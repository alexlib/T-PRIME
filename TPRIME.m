function [u_s, gradU_s, HessU_s, gradP_s, info] = TPRIME(xp, up, accel, xs, neighbors, varargin)
% Physics-informed interpolation using velocities + material acceleration (DU/Dt).
% Exact divergence constraint via Lagrange multipliers (KKT).
% Huber IRLS integrated with KKT so divergence is enforced to machine precision.
%
% GRADIENT CONVENTION:
%   gradU_s(c, d, i) = dU_c/dx_d   at target point xs(i)
%   Row c = gradient of velocity component c.
%
% QUADRATIC BASIS per component (cols 1..10):
%   [1,  x,  y,  z,  0.5x^2,  xy,  xz,  0.5y^2,  yz,  0.5z^2]
%    1   2   3   4     5       6    7      8        9    10
%   Gradients at origin: th(2)=dU/dx, th(3)=dU/dy, th(4)=dU/dz
%   Laplacian at origin: th(5)+th(8)+th(10)  

opts = inputParser;
opts.addParameter('Weights',            'gaussian');  % 'inv2'|'shepard'|'gaussian'
opts.addParameter('DivFree',            false);       % hard div-free constraint
opts.addParameter('LambdaVel',          1e-10);        % Tikhonov on velocity block
opts.addParameter('LambdaP',            1e-10);        % Tikhonov on pressure block
opts.addParameter('Rho',                1.0);         % density
opts.addParameter('Mu',                 0.0);         % dynamic viscosity
opts.addParameter('UsePressureHessian', false);       % include HessP terms
opts.addParameter('MomScale',           1.0);         % weight on momentum equations
opts.addParameter('BodyForce',          [0 0 0]);     % per-unit-mass body force
opts.addParameter('MinNeighbors',       10);
opts.addParameter('MaxCond',            1e9);
opts.addParameter('Verbose',            false);
opts.addParameter('Loss',               'L2');        % 'L2' or 'Huber'
opts.addParameter('HuberDelta',         1.0);
opts.addParameter('HuberDeltaAuto',     true);
opts.addParameter('AllowFallback',      false);  % allow neighbor expansion and fallback
opts.parse(varargin{:});

wmethod    = lower(opts.Results.Weights);
lambdaVel  = opts.Results.LambdaVel;
lambdaP    = opts.Results.LambdaP;
rho        = opts.Results.Rho;
mu         = opts.Results.Mu;
useHp      = opts.Results.UsePressureHessian;
momScale   = opts.Results.MomScale;
fvec       = opts.Results.BodyForce(:).';
minK       = opts.Results.MinNeighbors;
maxCond    = opts.Results.MaxCond;
vb         = opts.Results.Verbose;
divFree    = opts.Results.DivFree;
lossType   = lower(opts.Results.Loss);
huberDelta = opts.Results.HuberDelta;

% -------------------------------------------------------------------------
% Allocate outputs
% -------------------------------------------------------------------------
Ns = size(xs, 1);
u_s     = NaN(Ns, 3);
gradU_s = NaN(3, 3, Ns);
HessU_s = NaN(3, 3, 3, Ns);
gradP_s = NaN(Ns, 3);
HessP_s = NaN(3, 3, Ns);

info_Nused            = zeros(Ns, 1);
info_rankA            = NaN(Ns, 1);
info_condA            = NaN(Ns, 1);
info_resnorm          = NaN(Ns, 1);
info_flags            = strings(Ns, 1);
info_huber_iterations = zeros(Ns, 1);
info_n_outliers       = zeros(Ns, 1);
info_huber_delta      = zeros(Ns, 1);

% -------------------------------------------------------------------------
% Main loop
% -------------------------------------------------------------------------
parfor i = 1:Ns
    if isnan(xs(i,1)), continue; end

    idxAll = neighbors{i};
    if isempty(idxAll), continue; end
    % Preinitialize all variables that parfor warns about
    % w          = zeros(0,1);
    Phi10      = zeros(0,10);
    Phi4       = zeros(0,4);
    A_div      = zeros(1, 0);
    % ATA        = zeros(0,0);
    % theta      = zeros(0,1);
    huberDelta = opts.Results.HuberDelta;  % use the parsed value as default

    % Local copies of per-point variables
    % lv = lambdaVel;
    % lp = lambdaP;
    % 
    % triedAffine = false;
    % triedNoHp   = ~useHp;
    % flag        = "";


    k   = min(numel(idxAll), minK);
    idx = idxAll(1:k);

    triedAffine = false;
    triedNoHp   = ~useHp;
    flag        = "";

    lv = lambdaVel;
    lp = lambdaP;

    while true
        Xi = xp(idx,:);  Ui = up(idx,:);  Ai = accel(idx,:);
        m  = numel(idx);

        dX = Xi - xs(i,:);
        x  = dX(:,1);  y = dX(:,2);  z = dX(:,3);
        di = vecnorm(dX, 2, 2);

        % ------------------------------------------------------------------
        % Spatial weights
        % ------------------------------------------------------------------
        switch wmethod
            case 'inv2'
                w = 1 ./ max(di, 1e-12).^2;

            case 'shepard'
                r  = 2.0 * median(di + (di==0).*median(di));
                q  = di / max(r, eps);
                in = q < 1;
                w  = zeros(m, 1);
                if any(in)
                    w(in) = 1./max(di(in),1e-12).^2.5 .* (1-q(in)).^2;
                else
                    [~,ord] = sort(di);
                    w(ord(1:min(6,m))) = 1;
                end

            case 'gaussian'
                s     = std(di);
                sigma = max(s * m^(-1/(3+4)), median(di));  % Scott's rule
                w     = exp(-(di./sigma).^2);

            otherwise
                error('Unknown Weights: %s', wmethod);
        end
        w  = w / (sum(w) + eps);
        Wm = diag(sqrt(w));
        W3 = blkdiag(Wm, Wm, Wm);

        % ------------------------------------------------------------------
        % Model size
        % ------------------------------------------------------------------
        useAffineVel = triedAffine;

        if ~useAffineVel
            % Quadratic basis (m x 10) per component:
            % [1,  x,  y,  z,  0.5x^2, xy, xz, 0.5y^2, yz, 0.5z^2]
            %  1   2   3   4     5      6   7     8      9    10
            Phi10        = [ones(m,1), x, y, z, 0.5*x.^2, x.*y, x.*z, 0.5*y.^2, y.*z, 0.5*z.^2];
            nVelCols     = 30;
            basisPerComp = 10;
        else
            % Affine basis (m x 4): [1, x, y, z]
            Phi4         = [ones(m,1), x, y, z];
            nVelCols     = 12;
            basisPerComp = 4;
        end

        if ~triedNoHp && useHp
            nPcols = 9;
        else
            nPcols = 3;
        end
        nCols = nVelCols + nPcols;

        % ------------------------------------------------------------------
        % Velocity equations (3m rows)
        % ------------------------------------------------------------------
        A_vel = zeros(3*m, nVelCols);
        b_vel = zeros(3*m, 1);

        if ~useAffineVel
            for c = 1:3
                cols = (c-1)*10 + (1:10);
                A_vel((c-1)*m+1 : c*m, cols) = Phi10;
                b_vel((c-1)*m+1 : c*m)       = Ui(:,c);
            end
        else
            for c = 1:3
                cols = (c-1)*4 + (1:4);
                A_vel((c-1)*m+1 : c*m, cols) = Phi4;
                b_vel((c-1)*m+1 : c*m)       = Ui(:,c);
            end
        end

        % Spatially weighted velocity block (full nCols width)
        A_vel_full = zeros(3*m, nCols);
        A_vel_full(:, 1:nVelCols) = W3 * A_vel;
        b_vel_full = W3 * b_vel;

        % ------------------------------------------------------------------
        % Momentum equations (3m rows)
        %
        % Navier-Stokes: rho*DU/Dt = -gradP + mu*Lap(U) + rho*f
        % Rearranged:    -gradP + mu*Lap(U) - rho*accel + rho*f = 0
        %
        % Laplacian from quadratic basis (FIX 3):
        %   Lap(u) = d^2u/dx^2 + d^2u/dy^2 + d^2u/dz^2
        %          = th5*1    + th8*1    + th10*1
        %
        % Global Laplacian column indices per component:
        %   comp 1 (u): cols  5,  8, 10
        %   comp 2 (v): cols 15, 18, 20   (offset 10: 10+5, 10+8, 10+10)
        %   comp 3 (w): cols 25, 28, 30   (offset 20: 20+5, 20+8, 20+10)
        % ------------------------------------------------------------------
        A_mom = zeros(3*m, nCols);
        b_mom = zeros(3*m, 1);

        for krow = 1:m
            dx = dX(krow,:).';

            % x-momentum
            row = krow;
            A_mom(row, nVelCols+1) = -1;                       % -dp/dx
            if ~triedNoHp && useHp
                A_mom(row, nVelCols+4 : nVelCols+6) = -dx.';  % -Hp(1,:)*dx
            end
            if ~useAffineVel
                A_mom(row, [5 8 10]) = mu;                     % mu*Lap(u)
            end
            b_mom(row) = -rho*Ai(krow,1) + rho*fvec(1);

            % y-momentum
            row = m + krow;
            A_mom(row, nVelCols+2) = -1;                       % -dp/dy
            if ~triedNoHp && useHp
                A_mom(row, nVelCols+[5 7 8]) = -dx.';
            end
            if ~useAffineVel
                % cols: 10+5=15, 10+8=18, 10+10=20
                A_mom(row, 15) = A_mom(row,15) + mu;           % mu*Lap(v)
                A_mom(row, 18) = A_mom(row,18) + mu;
                A_mom(row, 20) = A_mom(row,20) + mu;
            end
            b_mom(row) = -rho*Ai(krow,2) + rho*fvec(2);

            % z-momentum
            row = 2*m + krow;
            A_mom(row, nVelCols+3) = -1;                       % -dp/dz
            if ~triedNoHp && useHp
                A_mom(row, nVelCols+[6 8 9]) = -dx.';
            end
            if ~useAffineVel
                % cols: 20+5=25, 20+8=28, 20+10=30
                A_mom(row, 25) = A_mom(row,25) + mu;           % mu*Lap(w)
                A_mom(row, 28) = A_mom(row,28) + mu;
                A_mom(row, 30) = A_mom(row,30) + mu;
            end
            b_mom(row) = -rho*Ai(krow,3) + rho*fvec(3);
        end

        % Spatially weighted + scaled momentum block
        A_mom_full = (momScale * W3) * A_mom;
        b_mom_full = (momScale * W3) * b_mom;

        % ------------------------------------------------------------------
        % Combined data system
        % ------------------------------------------------------------------
        Adata = [A_vel_full; A_mom_full];
        bdata = [b_vel_full; b_mom_full];
        % ------------------------------------------------------------------
        % Regularization
        % ------------------------------------------------------------------
        if lv > 0
            Rv = zeros(nVelCols, nCols);
            reg_diag = ones(nVelCols, 1) * sqrt(lv);
            
            if ~useAffineVel
                % Quadratic basis: selective regularization
                grad_cols = [2,3,4, 12,13,14, 22,23,24];
                quad_cols = [5:10, 15:20, 25:30];
                reg_diag(grad_cols) = sqrt(lv * 0.1);
                reg_diag(quad_cols) = sqrt(lv * 10);
            end
            % Affine basis: reg_diag stays uniform sqrt(lv) — only 12 elements, no quad terms
            
            Rv(:, 1:nVelCols) = diag(reg_diag);
        else
            Rv = zeros(0, nCols);
        end

        if lp > 0
            Rp = zeros(nPcols, nCols);
            Rp(:, nVelCols+1:nCols) = sqrt(lp) * eye(nPcols);
        else
            Rp = zeros(0, nCols);
        end

        Areg = [Rv; Rp];
        breg = zeros(size(Areg,1), 1);

        % ------------------------------------------------------------------
        % Divergence-free constraint operator
        % du/dx -> col 2 of comp-1 block  = col 0*basisPerComp+2
        % dv/dy -> col 3 of comp-2 block  = col 1*basisPerComp+3
        % dw/dz -> col 4 of comp-3 block  = col 2*basisPerComp+4
        % ------------------------------------------------------------------
        if divFree
            A_div    = zeros(1, nCols);
            div_idxs = [0*basisPerComp+2, 1*basisPerComp+3, 2*basisPerComp+4];
            if any(div_idxs > nCols) || any(div_idxs < 1)
                error('Divergence index out of range: [%s], nCols=%d', num2str(div_idxs), nCols);
            end
            A_div(div_idxs) = 1;
        end

        % ------------------------------------------------------------------
        % Auto-tune Huber delta from initial L2 residuals
        % Uses Adata (weighted) residuals — consistent with IRLS below
        % ------------------------------------------------------------------
        if strcmpi(lossType, 'huber') & opts.Results.HuberDeltaAuto
            if divFree
                ATA0 = Adata'*Adata + Areg'*Areg;
                ATb0 = Adata'*bdata + Areg'*breg;
                KKT0 = [ATA0, A_div'; A_div, 0];
                sol0 = KKT0 \ [ATb0; 0];
                th0  = sol0(1:nCols);
            else
                th0 = [Adata; Areg] \ [bdata; breg];
            end

            r0           = Adata * th0 - bdata;
            MAD          = median(abs(r0 - median(r0)));
            sigma_robust = 1.4826 * MAD;
            huberDelta   = max(1.5 * sigma_robust, 1e-12);

            if vb
                fprintf('Point %d: auto huberDelta=%.4e (sigma_robust=%.4e)\n', ...
                        i, huberDelta, sigma_robust);
            end
        end

        % ------------------------------------------------------------------
        % Solve
        % ------------------------------------------------------------------
        if divFree
            switch lossType
                case 'l2'
                    ATA   = Adata'*Adata + Areg'*Areg;
                    ATb   = Adata'*bdata + Areg'*breg;
                    KKT   = [ATA, A_div'; A_div, 0];
                    sol   = KKT \ [ATb; 0];
                    theta = sol(1:nCols);

                case 'huber'
                    nData  = size(Adata, 1);
                    wIRLS  = ones(nData, 1);
                    maxIter = 20;
                    tolIRLS = 1e-6;
                    theta   = zeros(nCols, 1);
                    ATA     = Adata'*Adata + Areg'*Areg;  % init for scope

                    for it = 1:maxIter
                        sw  = sqrt(wIRLS);
                        Aw  = bsxfun(@times, Adata, sw);
                        bw  = bdata .* sw;

                        ATA = Aw'*Aw + Areg'*Areg;
                        ATb = Aw'*bw + Areg'*breg;
                        KKT = [ATA, A_div'; A_div, 0];
                        sol = KKT \ [ATb; 0];
                        theta_new = sol(1:nCols);

                        r    = Adata * theta_new - bdata;
                        absr = abs(r);
                        wnew = ones(nData, 1);
                        large = absr > huberDelta;
                        wnew(large) = huberDelta ./ absr(large);

                        dtheta = norm(theta_new - theta) / (1 + norm(theta));
                        dw     = norm(wnew - wIRLS)      / (1 + norm(wIRLS));

                        if vb && mod(i,500)==0
                            fprintf('  [KKT] it%2d ||Δθ||=%.2e ||Δw||=%.2e out=%d\n', ...
                                    it, dtheta, dw, sum(large));
                        end

                        theta  = theta_new;
                        wIRLS  = wnew;
                        if dtheta < tolIRLS && dw < tolIRLS, break; end
                    end
                    info_huber_iterations(i) = it;
                    info_outliers(i) = sum(abs(Adata*theta - bdata) > huberDelta);

                otherwise
                    error('Unknown Loss: %s', lossType);
            end

            info_huber_delta(i) = huberDelta;
            A_soft = [Adata; Areg];
            b_soft = [bdata; breg];
            rs = norm(b_soft - A_soft*theta);
            rk = rank(ATA);
            cd = cond(ATA);

        else
            % divFree == false: stacked approach
            Aall = [Adata; Areg];
            ball = [bdata; breg];

            switch lossType
                case 'l2'
                    theta = Aall \ ball;

                case 'huber'
                    nData   = size(Adata, 1);
                    wIRLS   = ones(nData, 1);
                    maxIter = 20;
                    tolIRLS = 1e-6;
                    theta   = zeros(nCols, 1);

                    for it = 1:maxIter
                        sw  = sqrt(wIRLS);
                        Aw  = bsxfun(@times, Adata, sw);
                        bw  = bdata .* sw;

                        theta_new = [Aw; Areg] \ [bw; breg];

                        r    = Adata * theta_new - bdata;
                        absr = abs(r);
                        wnew = ones(nData, 1);
                        large = absr > huberDelta;
                        wnew(large) = huberDelta ./ absr(large);

                        dtheta = norm(theta_new - theta) / (1 + norm(theta));
                        dw     = norm(wnew - wIRLS)      / (1 + norm(wIRLS));

                        if vb && mod(i,500)==0
                            fprintf('  [stacked] it%2d ||Δθ||=%.2e ||Δw||=%.2e out=%d\n', ...
                                    it, dtheta, dw, sum(large));
                        end

                        theta  = theta_new;
                        wIRLS  = wnew;
                        if dtheta < tolIRLS && dw < tolIRLS, break; end
                    end
                    info_huber_iterations(i) = it;
                    info_n_outliers(i) = sum(abs(Adata*theta - bdata) > huberDelta);

                otherwise
                    error('Unknown Loss: %s', lossType);
            end

            info_huber_delta(i) = huberDelta;
            rs = norm(ball - Aall*theta);
            rk = rank(Aall);
            cd = cond(Aall);
        end

        % ------------------------------------------------------------------
        % Stability check & fallback
        % ------------------------------------------------------------------
        if rk < min(size(Adata,1)+size(Areg,1), nCols) || cd > maxCond
            if opts.Results.AllowFallback && numel(idx) < numel(idxAll)
                k   = min(numel(idxAll), numel(idx)+max(3,ceil(0.25*numel(idx))));
                idx = idxAll(1:k);
                if vb, fprintf('i=%d expanding to %d neighbors (rk=%d, cd=%.2e)\n',i,k,rk,cd); end
                continue;
            end
            if ~triedNoHp && useHp
                triedNoHp = true; flag = "NoHp";
                if vb, fprintf('i=%d disabling HessP (rk=%d, cd=%.2e)\n',i,rk,cd); end
                continue;
            end
            if ~triedAffine
                triedAffine = true; flag = flag + "+AffVel";
                if vb, fprintf('i=%d downgrading to affine vel (rk=%d, cd=%.2e)\n',i,rk,cd); end
                continue;
            end
            % lv = max(lv, 1e-4);
            % lp = max(lp, 1e-6);
            if vb, fprintf('i=%d boosted regularization; accepting.\n',i); end
        end

        % ------------------------------------------------------------------
        % Unpack
        % gradU_s(c, d, i) = dU_c/dx_d
        % ------------------------------------------------------------------
        if ~useAffineVel
            th1 = theta(1:10);  th2 = theta(11:20);  th3 = theta(21:30);
            [a1,b1,H1] = unpack_u(th1);
            [a2,b2,H2] = unpack_u(th2);
            [a3,b3,H3] = unpack_u(th3);
            u_s(i,:)         = [a1, a2, a3];
            gradU_s(:,:,i)   = [b1; b2; b3];   % row c = grad(U_c)
            HessU_s(:,:,:,i) = cat(3, H1, H2, H3);
        else
            th1 = theta(1:4);  th2 = theta(5:8);  th3 = theta(9:12);
            u_s(i,:)         = [th1(1), th2(1), th3(1)];
            gradU_s(:,:,i)   = [th1(2:4).'; th2(2:4).'; th3(2:4).'];
            HessU_s(:,:,:,i) = 0;
        end

        % Pressure
        gradP_s(i,:) = theta(nVelCols+1 : nVelCols+3).';

        info_Nused(i)   = m;
        info_rankA(i)   = rk;
        info_condA(i)   = cd;
        info_resnorm(i) = rs;
        info_flags(i)   = flag;

        break;
    end  
end  

info.Nused            = info_Nused;
info.rankA            = info_rankA;
info.condA            = info_condA;
info.resnorm          = info_resnorm;
info.flags            = info_flags;
info.huber_iterations = info_huber_iterations;
info.n_outliers       = info_n_outliers;
info.huber_delta      = info_huber_delta;

end  % main function
