function [H, alpha, Z, obj] = DSMVC(X, Z_ini, num_clust, lambda, beta, p, q, mu, max_iter)
% |Z|_sp^p + lambda|Z-HG'| + beta|H|_sp^p s.t. X{i} = X{i}*Z{i}

num_view = length(X);
Z = Z_ini;
N = size(Z{1}, 2);

sX = [N, num_clust, num_view];
eta = 1.3;
max_mu = 10e12;
betaf = ones(num_view, 1);
alpha = repmat(1 / num_view, [1, num_view]);

obj = zeros(max_iter, 1);

for i = 1:num_view
    H{i} = eye(N, num_clust);
    G{i} = H{i};
end

for v = 1:num_view
    S{v} = zeros(N, N);
    Y1{v} = zeros(N, num_clust);
    J{v} = zeros(N, num_clust);
    Y2{v} = zeros(N, num_clust);
    Y3{v} = zeros(N, N);
    Y4{v} = zeros(size(X{v}));
    Q{v} = zeros(N, num_clust);
end

for iter = 1:max_iter
    for i = 1:num_view
        A{i} = S{i} - Y3{i} ./ mu;
        B{i} = X{i} + Y4{i} ./ mu;
        Z{i} = inv(X{i}' * X{i} + eye(N, N)) * (A{i} + X{i}' * B{i});
    end

    H_hat = time2frequency(H);
    G_hat = time2frequency(G);

    for v = 1:num_view
        M1{v} = Z{v} + Y3{v} ./ mu;
        M2_hat{v} = H_hat{v} * G_hat{v}';
    end

    M1_hat = time2frequency(M1);

    for v = 1:num_view
        temp_hat{v} = (2 .* lambda .* M2_hat{v} + mu .* M1_hat{v}) ./ (2 .* lambda + mu);
    end

    temp = frequency2time(temp_hat);
    temp_tensor = cat(3, temp{:});
    sX1 = size(temp_tensor);
    temp_vector = temp_tensor(:);

    [mys, ~] = wshrinkObj_weight_lp(temp_vector, betaf ./ (2 .* lambda + mu), sX1, 0, 3, p);
    S_tensor = reshape(mys, sX1);

    for v = 1:num_view
        S{v} = S_tensor(:, :, v);
    end

    S_hat = time2frequency(S);

    for v = 1:num_view
        G_hat{v} = S_hat{v}' * H_hat{v};
    end

    G = frequency2time(G_hat);

    for v = 1:num_view
        temp1_hat{v} = S_hat{v} * G_hat{v};
    end

    temp1 = frequency2time(temp1_hat);

    for v = 1:num_view
        B{v} = 2 * temp1{v} + mu ./ lambda .* (Q{v} - Y1{v} ./ mu) ...
            + mu ./ lambda .* (J{v} - Y2{v} ./ mu);
    end

    B_hat = time2frequency(B);

    for v = 1:num_view
        [nn{v}, ~, vv{v}] = svd(B_hat{v}, 'econ');
        H_hat{v} = nn{v} * vv{v}';
    end

    H = frequency2time(H_hat);

    for v = 1:num_view
        Q{v} = H{v} + Y1{v} ./ mu;
        Q{v}(Q{v} < 0) = 0;
    end

    for v = 1:num_view
        M{v} = H{v} + Y2{v} ./ mu;
    end

    M_tensor = cat(3, M{:});
    M_vector = M_tensor(:);

    [myj, ~] = wshrinkObj_weight_lp(M_vector, beta * betaf ./ mu, sX, 0, 3, q);
    J_tensor = reshape(myj, sX);

    for v = 1:num_view
        J{v} = J_tensor(:, :, v);
    end

    for v = 1:num_view
        Y1{v} = Y1{v} + mu * (H{v} - Q{v});
        Y2{v} = Y2{v} + mu * (H{v} - J{v});
        Y3{v} = Y3{v} + mu * (Z{v} - S{v});
        Y4{v} = Y4{v} + mu * (X{v} - X{v} * Z{v});
    end

    mu = min(eta * mu, max_mu);

    obj(iter) = cal_tensorSp(Z, p) + beta * cal_tensorSp(H, q);
end

end