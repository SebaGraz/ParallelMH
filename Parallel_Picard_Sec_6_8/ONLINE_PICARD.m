function [count, tp, XX_out] = ONLINE_PICARD(X0, ZZ, UU, N, K, L, std_dev, nlogpost)
%ONLINE_PICARD_RWM  MATLAB translation of the updated code (with std_dev).
% V       : function handle, V(x) -> scalar
% X0      : initial state (row or column)
% K, N    : integers
% std_dev : scalar or 1xd / d-by-1 (elementwise scale for proposals)
% h       : scalar step-size

    % d  = numel(X0);
    [Nt, d] = size(ZZ);
    XX = zeros(Nt, d);
    Vt = nlogpost(exp(X0)) + sum(X0);
   

    X = repmat(X0, K+1, 1);       % (K+1) x d
    B = zeros(K, 1);
    lw = 1;
    up = lw + K - 1;
    [~, B, X, ~] = picard_map(Vt, X, ZZ(lw:up, :), UU(lw:up), B, L, std_dev, nlogpost);
    count = 0;

    p = tic;
    while true
        count = count + 1;
        count
        [gain, B, X, Vt] = picard_map(Vt, X, ZZ(lw:up, :), UU(lw:up), B, L, std_dev, nlogpost);

        XX(lw:lw+gain, :) = X(1:gain+1, :);
        lw = lw + gain;
        up = lw + K - 1;

        [X, B] = move_forward(X, B, gain);
        if lw >= N
            tp = toc(p);
            fprintf('count = %d\n', count);
            fprintf('elapsed time = %d\n', tp);
            XX_out    = XX(1:lw, :);
            return
        end
    end
end


function [gain, Bn, X, Vt] = picard_map(V0, X, Z, U, Bo, L, std_dev, NlogPost)
%PICARD_MAP  MATLAB translation

    [K, d] = size(Z);

    Bn = zeros(size(Bo));
    Xp = zeros(K, d);
    Vj = zeros(K, 1);

    Vtilde = V0;
    Vt     = V0;
    gain   = 0;
    first  = true;

    % make std_dev broadcast-friendly (supports scalar or 1xd)
   
    parfor (j = 1:K, 14)
        % Xp(j, :) = X(j, :) + (Z(j, :) .* sd_row) * h;
        Xp(j,:) = X(j, :) + (L/sqrt(d)*std_dev.*Z(j, :));
        Vj(j) = NlogPost(exp(Xp(j,:))) + sum(Xp(j,:));
    end

    for j = 1:K
        Bn(j) = (log(U(j)) < (Vtilde - Vj(j)));

        if Bo(j)
            Vtilde = Vj(j);
        end

        if first && (Bo(j) == Bn(j))
            gain = gain + 1;
            if Bo(j)
                Vt = Vj(j);
            end
        else
            first = false;
        end

        % X(j+1, :) = X(j, :) + double(Bn(j)) * (Z(j, :) .* sd_row) * h;
        X(j + 1, :) = X(j, :) + Bn(j) * L/sqrt(d)*std_dev.*Z(j, :);

    end
end


function [X, B] = move_forward(X, B, c)
%MOVE_FORWARD  MATLAB translation of move_forward!

    K = numel(B);

    X(1, :) = X(c + 1, :);

    % SHIFT
    if c < K
        for i = 1:(K - c)
            X(i + 1, :) = X(c + i + 1, :);
            B(i)        = B(c + i);
        end
    end

    % DRAW
    if c >= 1
        for i = (K - c + 1):K
            X(i + 1, :) = X(K - c + 1, :);
            B(i)        = false;
        end
    end
end
