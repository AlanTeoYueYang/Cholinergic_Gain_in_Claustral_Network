function X = ini_vec_revised(W)
    n = length(W);
    I = eye(n);
    Q = lyap(W-I, 2*I);
    [V,D] = eig(Q);
    [M,I] = max(diag(D));
    X = V(:,I);
    X = 1.5*sqrt(length(X))/norm(X)*X;
end