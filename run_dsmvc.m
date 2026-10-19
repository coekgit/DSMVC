clear;
clc

datasetname = {'MSRCV1_jie'};

for di = 1:length(datasetname)  % Datasets
    dataName = datasetname{di};
    load(dataName);
    disp(['\n\n Current dataset  : ',dataName]);
%     X = D;
%     load(Dataname);
    truth = Y;
    truthF = truth;
    numClust = length(unique(truthF));
    num_view = length(X);
    for iv = 1:num_view
        X1 = X{iv};
        X1 = NormalizeFea(X1,1);
        YY{iv} = X1';
    end
    clear X X1 W1
    X = YY;
    clear YY
    for iv = 1:num_view
        Z_ini{iv} = rand(size(X{iv},2));
    end

    para1 = [1e-3];
    para2 = [1];
    p1 = [0.8];
    p2 = [0.6];

    %     p = 0.9;
    mu = 1e-1;
    max_iter = 30;
    % bate = 100;
    for i=1:length(para1)
        for j=1:length(para2)
            for k = 1: length(p1)
                for l = 1:length(p2)
                    %             for l=1:length(para4)
                    [F,alpha,Z,obj] = DSMVC(X,Z_ini,numClust,para1(i),para2(j),p1(k),p2(l),mu,max_iter);
                    %             y = kmeans(H,numClust,'replicates', 10);
                    F_sum = F{1} / alpha(1);
                    for v = 2:num_view
                        F_sum = F_sum + F{v} / alpha(v);
                    end
                    alpha_sum = sum(1 ./ alpha);
                    F_final = F_sum / alpha_sum;
                    
                    disp('----------Clustering----------');
                    [~, Y_pre] = max(F_final, [], 2);
                    result_LatLRR = ClusteringMeasure(truthF, Y_pre)
                    ACC= result_LatLRR(1);
                    nmi= result_LatLRR(2);
                    Purity= result_LatLRR(3);
                    dlmwrite(['./', dataName, '_10_17_result.txt'],[para1(i) para2(j) p1(k) p2(l) ACC nmi Purity],'-append','delimiter','\t','newline','pc');
                end
                
            end
        end
    end
end
