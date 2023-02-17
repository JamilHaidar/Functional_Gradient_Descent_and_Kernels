clear;clc;close all;

tVec = -5:0.01:5;
% tVec = 0:0.01:1;

r = ones(1,length(tVec));
phi = ones(1,length(tVec));
% phi = 0.5*sin(tVec)+0.6*ones(1,length(tVec));

n_iter = 100;
eps = 0.01;
lr = 0.055;

% initial_condition_indices = [1 2];
initial_condition_indices = [];
for function_index = 1:size(r,1)
    for time_index = 1:length(initial_condition_indices)
        r(function_index,initial_condition_indices(function_index,time_index)) = solution_function(function_index,tVec(initial_condition_indices(function_index,time_index)));
    end
end
% r = linspace(r(1),r(end),length(tVec));
selected_indices = (1:length(tVec)).*ones(size(r,1),length(tVec));
selected_indices(initial_condition_indices) = [];
%%
figure;
for iter=0:n_iter
%     if iter==2000
%         lr = lr/10;
%     end
    if iter>0
%     dfdr = (cost_function(tVec,r+eps*perturb,gradient(r,tVec)+eps*gradient(perturb,tVec))-cost_function(tVec,r,gradient(r,tVec)))./eps;
        dfdr = functional_grad(tVec,r,phi,2,eps);
        r(selected_indices) = r(selected_indices) - lr*dfdr(selected_indices);
    end
    hold off;
    plot(tVec,r,'LineWidth',2)
    hold on;plot(tVec,solution_function(1,tVec))
    title(sprintf('Iteration: %d',iter))
    f{iter+1} = getframe(gcf);
    drawnow
%     pause(0.01)
end
%%
plot(tVec,r,'LineWidth',2)
hold on;plot(tVec,solution_function(1,tVec))
%% Create Animation
v = VideoWriter("sin.mp4",'MPEG-4');
open(v)
for iter=1:length(f)
    writeVideo(v,f{iter})
end
close(v)
%%
function g = functional_grad(t,r,phi,degree,eps)
    funcs = cell(size(r,1),1);
    perturbs = cell(size(r,1),1);
    for function_index = 1:size(r,1)
        funcs{function_index} = zeros(degree,length(t));
        funcs{function_index}(1,:) = r(function_index,:);
        perturbs{function_index} = zeros(degree,length(t));
        perturbs{function_index}(1,:) = phi(function_index,:);
        for deg=2:degree
            funcs{function_index}(deg,:) = gradient(funcs{function_index}(deg-1,:),t);
            perturbs{function_index}(deg,:) = gradient(perturbs{function_index}(deg-1,:),t);
        end
    end
    g = zeros(size(r,1),length(t));
    for function_index = 1:length(size(r,1))
        perturbed_function = funcs;
        perturbed_function{function_index} = perturbed_function{function_index}+eps*perturbs{function_index};
        g(function_index,:) = (cost_function(t,perturbed_function)-cost_function(t,funcs))./eps;
    end
end

function sol = solution_function(f_idx,t)
    f(1,:) = sin(t);
%     f(1,:) = exp(-t);
    sol = f(f_idx,:);
end

function f = cost_function(t,func)
    f = 0.5.*(func{1}(1,:)-sin(t)).^2;
%     f = 0.5.*(func{1}(1,:)+func{1}(2,:)).^2;
end