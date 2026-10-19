clear;
clc
addpath 'D:\coekgit\Script\MATLAB\Orth-NTF-main\funs'
datasetname = {'MSRCV1_jie'};

%% ========== 排列鲁棒性实验配置 ==========

perm_seeds = [42, 123, 456, 789, 1024];  % 加 -1 表示不打乱
num_perm_runs = length(perm_seeds);

all_results = cell(length(datasetname), 1);

for di = 1:length(datasetname)
    dataName = datasetname{di};
    load(dataName);
    disp(['\n\n Current dataset  : ', dataName]);
    
    truth = Y + 1;                    % handwritten: 0-9 -> 1-10
    truthF = truth;
    numClust = length(unique(truthF));
    num_view = length(X);
    
    % ---------- 数据整理 ----------
    for iv = 1:num_view
        X1 = X{iv};
        X1 = NormalizeFea(X1, 1);   % 当前不归一化
        YY{iv} = X1';                 % (d_v, N)
    end
    clear X X1 W1
    X = YY;
    clear YY
    
    % ---------- 参数 ----------
    para1 = [1e-3];
    para2 = [1e-1];
    p1 = [0.8];
    p2 = [0.6];
    mu = 1e-1;
    max_iter = 30;
    
    results_per_seed = zeros(num_perm_runs, 3);
    
    for si = 1:num_perm_runs
        for i = 1:length(p1)
            for j = 1:length(p2)
                rng(perm_seeds(si));
                n = size(X{1}, 2);
                
                % ========== 完全随机打乱（跨类混合） ==========
                if perm_seeds(si) == -1
                    perm = 1:n;                    % 不打乱
                else
                    perm = randperm(n);            % 完全随机打乱
                end
                
                inv_perm = 1:n;
                inv_perm(perm) = 1:n;              % 逆排列，用于恢复顺序
                
                % 同步打乱所有视图
                X_perm = cell(1, num_view);
                for iv = 1:num_view
                    X_perm{iv} = X{iv}(:, perm);
                end
                
                disp(['  >> Permutation run #', num2str(si), ...
                    ' (seed=', num2str(perm_seeds(si)), ')']);
                
                % Z_ini 初始化（用同一个 rng 状态）
                for iv = 1:num_view
                    Z_ini_perm{iv} = rand(size(X_perm{iv}, 2));
                end
                
                % 跑算法
                [F, alpha, Z, obj] = ...
                    low_rank_tensor_5_dbg(X_perm, Z_ini_perm, numClust, ...
                    para1, para2, p1(i), p2(j), mu, max_iter);
                
                % ---------- 后处理 ----------
                F_sum = F{1} / alpha(1);
                for v = 2:num_view
                    F_sum = F_sum + F{v} / alpha(v);
                end
                alpha_sum = sum(1 ./ alpha);
                F_final = F_sum / alpha_sum;
                
                % 恢复到原始样本顺序
                F_final = F_final(inv_perm, :);
                
                % ---------- 评估 ----------
                disp('----------Clustering----------');
                [~, Y_pre] = max(F_final, [], 2);
                result_LatLRR = ClusteringMeasure(truthF, Y_pre);
                ACC    = result_LatLRR(1);
                nmi    = result_LatLRR(2);
                Purity = result_LatLRR(3);
                
                results_per_seed(si, :) = [ACC, nmi, Purity];
                
                dlmwrite(['./', dataName, '_perm_', num2str(perm_seeds(si)), ...
                    '_result_10_18.txt'], ...
                    [perm_seeds(si) para1 para2 p1(i) p2(j) ACC nmi Purity], ...
                    '-append', 'delimiter', '\t', 'newline', 'pc');
            end
        end
    end
    
    % ---------- 汇总统计 ----------
    mean_res = mean(results_per_seed, 1);
    std_res  = std(results_per_seed, 0, 1);
    
    disp('==============================================');
    disp(['Dataset: ', dataName]);
    disp(['  Mean ACC  = ', num2str(mean_res(1)), ...
        ' ± ', num2str(std_res(1))]);
    disp(['  Mean NMI  = ', num2str(mean_res(2)), ...
        ' ± ', num2str(std_res(2))]);
    disp(['  Mean Pur. = ', num2str(mean_res(3)), ...
        ' ± ', num2str(std_res(3))]);
    disp('==============================================');
    
    all_results{di} = [mean_res; std_res];
    dlmwrite(['./', dataName, '_perm_summary_10_18.txt'], ...
             [di, mean_res, std_res], ...
             'delimiter', '\t', 'newline', 'pc');
end