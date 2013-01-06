%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A single layer perceptron example - implementing an OR gate
% Gopi(dot)Subramanian(at)gmail.com
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
source("./common/Activations.m");

% Input vector
X =[0,0;0,1;1,0;1,1];
y =[0;1;1;1];

% Add bias to input
% X is now a 4 * 3 matrix
X =[ones(size(X,1),1) X];

% Weight is a 3 * 1 matrix
% Intialize it with random weights
W = normrnd(0,eps^2,size(X,2),1);
iterations =10;
% Learning rate
lambda =0.001;
 
for i=1:iterations
	% Feed forward
	A=sigmoid(X*W);

	j = find(A <= 0.5);
	A(j) =0;
	j = find(A > 0.5);
	A(j) =1;
	direction = y-A;
	
	fprintf("Iteration %d \n",i);
	fprintf("	Direction \n");
	fprintf("		%f \n",direction);
	
	fprintf("	Activation \n");
	fprintf("		%f \n",A);

	fprintf("	Weight \n");
	fprintf("		%f \n",W);
	
	
	% Update the weights
	W = W + lambda .* (direction' * X)';
end

% Test Case
fprintf("Test for 1,0 \n");
X_test=[1,1,0;];  
A_test = sigmoid(X_test*W);
j = find(A_test <= 0.5);
A_test(j) =0;
j = find(A_test > 0.5);
A_test(j) =1;

fprintf(" Test output %d \n",A_test);
	

