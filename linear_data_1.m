clear all;
close all;
clc;

randn('state', 1);

sigmal = 1;
sigmae = 0.05;
N=100;

%generate quadratic data
a = sigmal*randn();b = sigmal*randn();
x = randn(N,1);
A = [x,ones(N,1)];
e = sigmae*randn(N,1);
y = A*[b;a] + e;

plot(x,y,'.')

writetable(table(x,y),'linear_data.csv')
