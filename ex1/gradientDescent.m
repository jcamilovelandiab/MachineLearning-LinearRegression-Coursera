function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	hTheta = X*theta;
	s1 = 0; i=1;
	for i=1:m
		s1 = s1 + hTheta(i)-y(i);
	end
	v1 = theta(1)-alpha*(1/m)*s1;
	
	s2 = 0; i=1;
	for i=1:m
		s2 = s2 + (hTheta(i)-y(i))*X(i,2);
	end
	
	v2 = theta(2) - alpha*(1/m)*s2;
	
	theta = [v1;v2];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
