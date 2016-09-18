function [all_theta] = oneVsAll(X, y, num_labels, lambda)

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

theta = zeros(n+1,1);

initial_theta = zeros(n+1,1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 50);
   
for current_label = 1:num_labels

   y_dash = y==current_label;

   % Optimize
   [theta, J, exit_flag] = fminunc(@(t)(lrCostFunction(t, X, y_dash, lambda)), initial_theta, options);
    
   all_theta(current_label,:) = theta';

endfor

end
