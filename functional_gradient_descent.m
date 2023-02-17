clear;clc;close all;
tVec = -pi:0.01:pi;
% tVec = 0:0.01:2;
r = ones(1,length(tVec));
% phi = ones(1,length(tVec));
% phi = 0.5*sin(tVec)+0.6*ones(1,length(tVec));
degree = 1;

n_iter = 1000;
eps = 0.001;
% eps = 1;
lr = 0.8;

% initial_condition_indices = [1 length(tVec)];
initial_condition_indices = [];
for function_index = 1:size(r,1)
    for time_index = 1:length(initial_condition_indices)
        r(function_index,initial_condition_indices(function_index,time_index)) = solution_function(function_index,tVec(initial_condition_indices(function_index,time_index)));
    end
end
selected_indices = (1:length(tVec)).*ones(size(r,1),length(tVec));
selected_indices(initial_condition_indices) = [];
%%
figure;
for iter=0:n_iter
    if iter==1001
        lr = lr/10;
    end
    if iter>0
%     dfdr = (cost_function(tVec,r+eps*perturb,gradient(r,tVec)+eps*gradient(perturb,tVec))-cost_function(tVec,r,gradient(r,tVec)))./eps;
        dfdr = functional_grad(tVec,r,degree,eps);
        r(selected_indices) = r(selected_indices) - lr*dfdr(selected_indices);
    end
    hold off;
    plot(tVec,r,'LineWidth',2)
    hold on;plot(tVec,solution_function(1,tVec))
    title(sprintf('Iteration: %d',iter))
    f{iter+1} = getframe(gcf);
    drawnow
end
%%
plot(tVec,r,'LineWidth',2)
% hold on;plot(tVec,exp(-tVec))
hold on;plot(tVec,solution_function(1,tVec))
%%
function g = functional_grad(t,r,degree,eps)
    funcs = cell(size(r,1),1);
    dt = t(2)-t(1);
    perturbs = zeros(degree,length(t),length(t));
    for function_index = 1:size(r,1)
        funcs{function_index} = zeros(degree,length(t),length(t));
        funcs{function_index}(1,:,:) = repmat(r(function_index,:),length(t),1);
        perturbs(1,:,:) = eye(length(t));
        for deg=2:degree
            funcs{function_index}(deg,:,:) = repmat(gradient(reshape(funcs{function_index}(deg-1,1,:),1,[]),t),length(t),1);
            perturbs(deg,:,:) = gradient(squeeze(perturbs(deg-1,:,:)))./dt;
        end
    end
    g = zeros(size(r,1),length(t));
    tt = repmat(t,length(t),1);
    for function_index = 1:length(size(r,1))
        perturbed_function = funcs;
        perturbed_function{function_index} = perturbed_function{function_index}+eps*perturbs;
        Fp  = trapz(squeeze(cost_function(tt,perturbed_function)),2).*dt;
        F = trapz(squeeze(cost_function(tt,funcs)),2).*dt;
        g(function_index,:) = (Fp-F)'./eps;
    end
end

function sol = solution_function(f_idx,t)
    f(1,:) = sin(t);
%     f(1,:) = exp(-t);
    sol = f(f_idx,:);
end

function f = cost_function(t,func)
    f = 0.5.*(squeeze(func{1}(1,:,:))-sin(t)).^2;
%     f = 0.5.*(func{1}(1,:,:)+func{1}(2,:,:)).^2;
end