close all;
clc;

randn('state', 1);

sigmal = 1;
sigmae = 0.50;
N=100;

%generate quadratic data
a = sigmal*randn();b = sigmal*randn();c = sigmal*randn();
x = randn(N,1);
A = [x.^2,x,ones(N,1)];
e = sigmae*randn(N,1);
y = A*[c;b;a] + e;

plot(x,y,'.')
[T,L_X,L_Y] = table(x,y)
%writetable(table(x,y),'quadratic_data.csv')
