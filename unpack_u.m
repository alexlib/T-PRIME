% -------- helper to unpack velocity parameters ----------
function [a,b,H] = unpack_u(theta)
% theta = [a, bx,by,bz, 0.5xx, xy, xz, 0.5yy, yz, 0.5zz]
a = theta(1);
b = theta(2:4).';
H = [theta(5) theta(6) theta(7);
     theta(6) theta(8) theta(9);
     theta(7) theta(9) theta(10)] * 2;  % undo 0.5
end