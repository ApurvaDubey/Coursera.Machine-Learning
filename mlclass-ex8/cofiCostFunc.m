function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies,num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% [J, grad] = cofiCostFunc(params=[sin(1:12)';cos(1:10)'], Y=reshape(sin(1:30), 6, 5), R=(reshape(sin(1:30), 6, 5)>0.5), n_u=5, n_m=6,n=2, lambda=0)
% 
% J =  8.3660
% grad =
% 
%    0.34484
%    3.75643
%    1.33604
%    0.00000
%    0.00000
%    0.00000
%   -0.15161
%    3.65262
%    2.37117
%    0.00000
%    0.00000
%    0.00000
%    0.68902
%   -0.99619
%   -1.97015
%   -2.37433
%   -1.39181
%    0.68659
%   -0.95155
%   -2.36339
%   -2.92067
%   -1.84072

% [J, grad] = cofiCostFunc(params=[sin(1:12)';cos(1:10)'], Y=reshape(sin(1:30), 6, 5), R=(reshape(sin(1:30), 6, 5)>0.5), n_u=5, n_m=6,n=2, lambda=2)
% 
% J =  19.654
% grad =
% 
%    2.02778
%    5.57503
%    1.61828
%   -1.51360
%   -1.91785
%   -0.55883
%    1.16236
%    5.63134
%    3.19540
%   -1.08804
%   -1.99998
%   -1.07315
%    1.76962
%   -1.82848
%   -3.95013
%   -3.68162
%   -0.82449
%    2.60693
%    0.55626
%   -2.65439
%   -4.74293
%   -3.51886


% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end),num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


J =  sum(sum(((X*Theta' .- Y) .* R).^2))/2 + (lambda/2)*sum(sum(Theta.^2)) + (lambda/2)*sum(sum(X.^2));

X_grad = ((X*Theta' .- Y) .* R)*Theta + lambda*X;

Theta_grad = ((X*Theta' .- Y) .* R)'*X + + lambda*Theta;



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
