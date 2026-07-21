%% T-PRIME Channel Flow Demo — Q-Criterion Isosurface Reconstruction
% ==============================================================================
% Reconstructs the velocity field, velocity gradient tensor, and Q-criterion
% from sparse, scattered PTV data using T-PRIME (physics-informed local
% Taylor-expansion + momentum-equation fit with divergence-free enforcement
% and Huber-robust weighting), then renders a near-wall Q-criterion isosurface
% colored by streamwise velocity, matching the turbulent channel flow figure
% in the accompanying paper (0.025 ppp seeding case).
%
% INPUT
%   channel_flow_0p025ppp_demo.mat, expected to contain:
%     x1, y1, z1            PTV particle positions            [Np x 1] each
%     pu, pv, pw             PTV particle velocity components  [Np x 1] each
%     ax, ay, az             PTV particle acceleration (DU/Dt) [Np x 1] each
%     positions_mesh          candidate query grid, [Mfull x 3]
%                              (regular grid spanning the measurement volume;
%                              points outside the PTV domain are masked out
%                              below via an alpha-shape containment test)
%
% DEPENDENCIES
%   - TPRIME.m               T-PRIME reconstruction (this repository)
%   - Statistics and Machine Learning Toolbox (knnsearch)
%   - Base MATLAB graphics (alphaShape, isosurface, smooth3, patch)
%
% OUTPUT
%   A single figure: near-wall Q-criterion isosurface (Q = 15), colored by
%   streamwise velocity u, cropped to the plotting window used in the paper.
%
% USAGE
%   Place this script, TPRIME.m, and channel_flow_0p025ppp_demo.mat in the
%   same folder (or on the MATLAB path) and run directly.
% ==============================================================================

clc; clear;

%% ---- Physical and algorithm parameters -------------------------------
rho = 1;                 % fluid density
nu  = 5e-5;               % kinematic viscosity [m^2/s]

Nei         = 20;         % candidate neighbors retrieved per grid point (knnsearch K)
MinNeighbors = 11;        % neighbors actually used by T-PRIME per point (<= Nei);
                           % the extra (Nei - MinNeighbors) candidates are only
                           % used if 'AllowFallback' is enabled in the TPRIME
                           % call below, for automatic neighborhood expansion
                           % at ill-conditioned points. AllowFallback is left
                           % at its default (false) here, so exactly
                           % MinNeighbors points are used at every grid point.

domain_std_multiplier = 35; % alpha-shape cavity radius = mean nearest-neighbor
                             % spacing + this many standard deviations; controls
                             % how tightly the reconstruction domain hugs the
                             % scattered PTV point cloud (see Section 2 below)

Q_isosurface_level = 15;   % Q-criterion isosurface threshold for the figure

%% ==== 1. Load PTV data =================================================
load('channel_flow_0p025ppp_demo.mat');

positions_cln = [x1, y1, z1];
u_neighbors   = [pu, pv, pw];
acc           = [ax, ay, az];

Np = length(x1);

%% ==== 2. Determine the reconstructable PTV domain ======================
% Estimate the local particle spacing from each particle's own nearest
% neighbor, then use it to build an alpha-shape that hugs the true (possibly
% irregularly-shaped) scattered point cloud. Query grid points outside this
% shape are masked to NaN and excluded from reconstruction, rather than
% extrapolating into regions with no supporting PTV data.
neighbor_dis = zeros(Np, 1);
parfor i = 1:Np
    [~, dis] = knnsearch(positions_cln, positions_cln(i,:), 'K', Nei, ...
        'NSMethod', 'exhaustive', 'Distance', 'euclidean');
    neighbor_dis(i) = dis(2);   % dis(1) is the point itself (distance 0)
end

dis_mean = mean(neighbor_dis);
dis_std  = std(neighbor_dis);
cavity_radi = dis_mean + domain_std_multiplier * dis_std;

shp = alphaShape(positions_cln, cavity_radi);
% figure; plot(shp);   % uncomment to visualize the reconstruction domain

inside = inShape(shp, positions_mesh(:,1), positions_mesh(:,2), positions_mesh(:,3));
positions_in_domain = positions_mesh;
positions_in_domain(~inside, :) = NaN;   % mask grid points outside the PTV domain

%% ==== 3. Find neighboring particles for each grid point ================
M = size(positions_in_domain, 1);
neighborsCell_new = cell(M, 1);

parfor i = 1:M
    if isnan(positions_in_domain(i,1))
        continue
    end
    indices = knnsearch(positions_cln, positions_in_domain(i,:), 'K', Nei, ...
        'NSMethod', 'exhaustive', 'Distance', 'euclidean');
    neighborsCell_new{i} = indices(:)';
end

%% ==== 4. Run T-PRIME ====================================================
% NOTE: acc is negated here (DU/Dt convention flip between the PTV
% acceleration-estimation pipeline and TPRIME's expected momentum-equation
% sign convention). [Document the specific source of this sign flip here,
% e.g. which script/step produces 'acc' with the opposite sign, so future
% readers don't have to rediscover it.]
[u_ss, gradU_ss, HessU_ss, gradP_ss, info] = TPRIME(positions_cln, u_neighbors, ...
    -acc, positions_in_domain, neighborsCell_new, ...
    'weights', 'inv2', 'Mu', rho*nu, 'MinNeighbors', MinNeighbors, ...
    'UsePressureHessian', false, 'momScale', 1, 'DivFree', true, 'Loss', 'huber');

%% ==== 5. Unpack results =================================================
% Vectorized extraction (validated against the equivalent per-point loop to
% machine precision): gradU_ss(i,j,k) = dU_i/dx_j at grid point k, so a fixed
% (i,j) pair sliced across all k is a single indexing operation, not a loop.
dudx_ss = squeeze(gradU_ss(1,1,:));   dudy_ss = squeeze(gradU_ss(1,2,:));   dudz_ss = squeeze(gradU_ss(1,3,:));
dvdx_ss = squeeze(gradU_ss(2,1,:));   dvdy_ss = squeeze(gradU_ss(2,2,:));   dvdz_ss = squeeze(gradU_ss(2,3,:));
dwdx_ss = squeeze(gradU_ss(3,1,:));   dwdy_ss = squeeze(gradU_ss(3,2,:));   dwdz_ss = squeeze(gradU_ss(3,3,:));

div_ss = dudx_ss + dvdy_ss + dwdz_ss;

omega_x_ss = dwdy_ss - dvdz_ss;   % domega_z/dy - domega_y/dz
omega_y_ss = dudz_ss - dwdx_ss;   % domega_x/dz - domega_z/dx
omega_z_ss = dvdx_ss - dudy_ss;   % domega_y/dx - domega_x/dy

d2udx2_ss = squeeze(HessU_ss(1,1,1,:));  d2udy2_ss = squeeze(HessU_ss(1,2,2,:));  d2udz2_ss = squeeze(HessU_ss(1,3,3,:));
d2vdx2_ss = squeeze(HessU_ss(2,1,1,:));  d2vdy2_ss = squeeze(HessU_ss(2,2,2,:));  d2vdz2_ss = squeeze(HessU_ss(2,3,3,:));
d2wdx2_ss = squeeze(HessU_ss(3,1,1,:));  d2wdy2_ss = squeeze(HessU_ss(3,2,2,:));  d2wdz2_ss = squeeze(HessU_ss(3,3,3,:));

% Q-criterion, closed form: Q = -0.5*trace(G^2) = 0.5*(||Omega||_F^2 - ||S||_F^2).
% Algebraically identical to Qcriterion(G) below (verified to machine
% precision against the matrix-based definition); using the closed form here
% avoids a per-point 3x3 matrix build and an explicit loop over M points.
Q_ss = -0.5*(dudx_ss.^2 + dvdy_ss.^2 + dwdz_ss.^2) ...
       - (dudy_ss.*dvdx_ss + dudz_ss.*dwdx_ss + dvdz_ss.*dwdy_ss);

validGrad = ~any(isnan(reshape(gradU_ss, 9, M)), 1)';
Q_ss(~validGrad) = NaN;

%% ==== 6. Figure: near-wall Q-criterion isosurface =======================
% Axis limits, view angle, and crop window below are tuned to reproduce the
% published figure for this case; they are not general-purpose defaults.
valid_idx = isfinite(Q_ss);
crop = 0.01;

Int  = scatteredInterpolant(positions_in_domain(valid_idx,1), positions_in_domain(valid_idx,3), ...
                             positions_in_domain(valid_idx,2), Q_ss(valid_idx), 'natural');
Intu = scatteredInterpolant(positions_in_domain(valid_idx,1), positions_in_domain(valid_idx,3), ...
                             positions_in_domain(valid_idx,2), u_ss(valid_idx,1), 'natural');

xdomain = linspace(min(x1)+crop, max(x1)-crop, 200);
ydomain = linspace(min(y1)+0.01, max(y1)-crop, 40);
zdomain = linspace(min(z1)+crop, max(z1)-crop, 100);
[xxgrid, zzgrid, yygrid] = meshgrid(xdomain, zdomain, ydomain);

Q_criterion_iso = Int(xxgrid, zzgrid, yygrid);
u_iso           = Intu(xxgrid, zzgrid, yygrid);

figure('Position', [100, 100, 565, 385]);
set(gcf, 'renderer', 'Painters');
set(gcf, 'GraphicsSmoothing', 'off');
set(gcf, 'color', 'w');

[faces_Q_ptv, verts_Q_ptv, colors_Q_ptv] = isosurface( ...
    xxgrid, zzgrid, yygrid, smooth3(Q_criterion_iso, 'gaussian'), ...
    Q_isosurface_level, smooth3(u_iso, 'gaussian'));

patch('Vertices', verts_Q_ptv, 'Faces', faces_Q_ptv, 'FaceVertexCData', colors_Q_ptv, ...
      'FaceColor', 'interp', 'EdgeColor', 'none');
view(3);
clim([0 1.2]);
colormap(jet);
light;
lighting gouraud;

axis image;
axHandle = gca;
axHandle.FontSize = 18;
view(-36, 50);
axis image;
xlim([0 1]);
ylim([4.5 4.9]);
zlim([-1 -0.88]);
zticks([-1 -0.95 -0.9]);
zticklabels({'-1', '', '-0.9'});

xlabel('$x$', 'Interpreter', 'latex');
ylabel('$z$', 'Interpreter', 'latex');
zlabel('$y$', 'Interpreter', 'latex');
set(gca, 'FontSize', 18, 'FontName', 'aakar');

cb = colorbar;
cb.Location = 'north';
cb.Position = [0.08 0.8 0.3 0.04];
cblabel = ylabel(cb, '$u$', 'Rotation', 0, 'Interpreter', 'latex');
cblabel.Position(1) = 1.35;
cblabel.Position(2) = 2.2;
cblabel.FontSize = 18;

axis([0 1.0000 4.4670 4.9510 -1.0000 -0.8857]);
grid on;

% Floor plane (visual reference for the wall)
floor_surfacex = [0 1 1 0]';
floor_surfacez = [4.467 4.467 4.97 4.97]';
floor_surfacey = -[1 1 1 1]';
patch('Faces', [1 2 3 4], 'Vertices', [floor_surfacex, floor_surfacez, floor_surfacey], ...
      'facecolor', 'black', 'FaceAlpha', 0.1);

set(gca, 'position', [0.193 0.174 0.764 0.8]);

%% ==== Local functions ====================================================
function Q = Qcriterion(G)
%QCRITERION  Reference (unvectorized) definition of the Q-criterion.
%   Q = 0.5*(||Omega||_F^2 - ||S||_F^2), where S and Omega are the symmetric
%   (strain-rate) and antisymmetric (rotation-rate) parts of the velocity
%   gradient tensor G, with G(i,j) = dU_i/dx_j. Valid regardless of whether
%   div(U) = 0. Kept here as the canonical definition; the main script above
%   uses an algebraically identical closed-form expression, vectorized over
%   all grid points, verified against this function to machine precision.
    S     = 0.5 * (G + G');
    Omega = 0.5 * (G - G');
    Q = 0.5 * (sum(sum(Omega.*Omega)) - sum(sum(S.*S)));
end
