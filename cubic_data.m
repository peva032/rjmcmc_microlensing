clear all;
close all;
clc;

randn('state', 1);

sigmal = 1;
sigmae = 0.05;
N=100;

%generate cubic data
a = sigmal*randn();b = sigmal*randn();c = sigmal*randn();d = sigmal*randn();
x = randn(N,1);
A = [x.^3,x.^2,x,ones(N,1)];
e = sigmae*randn(N,1);
y = A*[d;c;b;a] + e;

plot(x,y,'.')
writetable(table(x,y),'cubic_data.csv')
