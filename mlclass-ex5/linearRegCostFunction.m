function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0; J1 = 0; J2 = 0;
grad = zeros(size(theta));

% Cost
z = [X * theta] - y;
J1 = sum(z .* z)/(2*m);
J2 = (lambda/(2*m))*sum(theta(2:end) .^ 2);
J = J1 + J2;

% Gradient

tmp_1 = (1/m) * X(:,1)' * ([X * theta] - y);

tmp_2 = (1/m) * X(:,2:end)' * ([X * theta] - y) + [lambda/m]*(theta(2:end));

grad = [tmp_1 ; tmp_2];


end
