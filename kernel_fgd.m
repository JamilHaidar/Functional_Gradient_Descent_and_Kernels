%% Initialize
clear;clc;close all;
learning_rate = 0.01;
lambda_reg = 0.01; % Regularization coefficient
n_iter = 200;
seed = 0;
rng(seed);

%% Create Dataset

x = linspace(-5,5,100)';
y = exp(-(((x - 0.5)./0.5) .^ 2)) + exp(-(((x + 0.5)./0.5) .^ 2));
% y = sin(-x);
% figure;plot(x,y);ylim([min(y)-1,max(y)+1]);

%% Create Kernel
% This kernel represents all function evaluations centered on each point xc
kernel_matrix = zeros(length(x),length(x));
for i = 1:length(x)
    for j = 1:length(x)
        kernel_matrix(i,j) = rbf_kernel(x(i),x(j),4);
    end
end

%% Perform descent

alpha = rand(size(x));
fx = zeros(length(x),n_iter);
for iter = 1:n_iter
    fx(:,iter) = kernel_matrix*alpha;
    alpha = 2 * learning_rate * (y - fx(:,iter)) + (1 - 2 * lambda_reg * learning_rate) * alpha;
end

%% Animate descent
figure;
for iter=1:n_iter
    hold off;
    plot(x,y);hold on;
    plot(x,fx(:,iter));
    title(sprintf('Iteration: %d',iter))
    f{iter} = getframe(gcf);
%     ylim([0 max([max(y) max(max(fx))])]);
    pause(0.01)
end
%% Create Animation
v = VideoWriter("rbf_kernel.mp4",'MPEG-4');
open(v)
for iter=1:length(f)
    writeVideo(v,f{iter})
end
close(v)
%% Functions

function k = rbf_kernel(x1,x2,gamma)
    k = exp(-gamma*(x1 - x2)^2);
end