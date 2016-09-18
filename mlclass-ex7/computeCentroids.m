function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);
count = zeros(K);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for i = 1:m % loop through all examples in X
	
	for k = 1:K % loop through all centroids
	
		if idx(i) == k
		
			count(k) = count(k) + 1;
		
			for j = 1:n % 'n' determines number of columns in centroid, we calculate average by entriod
			
				centroids(k, j) = centroids(k, j)  +  X(i,j);
				
			end
			
		end
		
	end
	
end

	
for p = 1:K
	
	centroids(k,:)  = centroids(k,:)/count(k);
	
end

% =============================================================


end

