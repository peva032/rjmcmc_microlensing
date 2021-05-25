%=======================================================================
%                       binary_light_curve
%
%This script solves the binary lens equation for a given source position.
%Eventually it will also generate a light curve for a given source
%position vector.
%=======================================================================

zs = 0.5+0.5*i;

z1 = -0.1;
z2 = 0.8;
m1 = 0.9;
m2 = 0.1;
zs1 = conj(zs) - z1;  %not sure if the complex conjugate of zs matters here.
zs2 = conj(zs) - z2;
zsa = 2.*conj(zs) - z1 - z2;
zsm = zs1.*zs2;
a = -(z1+z2);
b = z1*z2;

k = [1,2*a*b,(2*b+a^2),2*a,1];
g = [k(1)*zsm,b*zsa+k(2)*zsm,1+a*zsa+k(3)*zsm,k(4)*zsm+zsa,k(5)*zsm];

%Coefficients for 5th order polynomial.
c = [-g(1)-k(1).*conj(zs),g(1)-g(2).*zs-k(2).*conj(zs)-b,g(2)-g(3).*zs-k(3).*conj(zs)-a,g(3)-g(4).*zs-k(4).*conj(zs)-1,g(4)...
-g(5).*zs-k(5).*conj(zs),g(5)];

%solve for roots to the polynomial
C = fliplr(c); %Flipping the order left to right
Z = roots(C)

%check that the roots satisfy the lens equation
%root = Z((zs - Z - m1./(conj(z1)-conj(Z)) - m2./(conj(z2)-conj(Z)))==0)
checked_roots = zs - Z - m1./(conj(z1)-conj(Z)) - m2./(conj(z2)-conj(Z))
% cz1 = conj(z1);
% cz2 = conj(z2);
% cz = conj(root);
%
% part1 = cz./((cz-cz1).*(cz-cz2).^2)-1./((cz-cz1).*(cz-cz2)) + cz./((cz-cz2).*(cz-cz1).^2);
% part2 = root./((root-z1).*(root-z2).^2) - 1./((root-z1).*(root-z2)) + cz./((root-z2).*(root-z1).^2);
%
% %Computing the amplification of the light curve A
% det_J = 1 - part1.*part2;
% A_i = a./det_J;
% A = sum(A_i);
