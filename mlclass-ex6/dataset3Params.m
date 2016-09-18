function [C, sigma, error_master] = dataset3Params(X, y, Xval, yval)
% X = [sin(1:1000)' cos(1:1000)']
% y =  floor(abs(sin(1:1000) + 0.5))'
% Xval = [cos(1:1000)' sin(1:1000)']
% yval = floor(abs(cos(1:1000) + 0.8))'

%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% With C & sigma vectors defined as below:
% 	C_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% 	sigma_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% [C, sigma] = dataset3Params([sin(1:3)'  cos(1:3)'], [1 0 1]', [sin(1:3)'  cos(1:3)'], [1 0 1]')
% C =  0.010000
% sigma =  0.010000

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

x1 = X(:,1);
x2 = X(:,2);

%C_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%sigma_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

C_vect = [0.01; 0.1;1;10;100;1000] ;
sigma_vect = [0.1];


error_master = [];

for i = 1:length(C_vect)
	for j = 1:length(sigma_vect)
				
		model = svmTrain(X, y, C_vect(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vect(j))); 
		
		predictions = svmPredict(model, Xval);
		
		err = mean(double(predictions ~= yval));
		
		error_master = [error_master ; C_vect(i) sigma_vect(j) err];
		
	end
end

[tmp , ix] = min(error_master(:,3));

C = error_master(ix,1);

sigma = error_master(ix,2);


% =========================================================================

end
