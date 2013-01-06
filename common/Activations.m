%%
% Activation functions
%
%
%

function g = linear(W,X)
	g = W * X;
end

function g = sigmoid(z)
	% Initalize the return value
	g=1 ./ (1+ exp(-z));
end

function g=tanh(z)
	g=zeros(size(z));
	g=exp(-z) + exp(z) ./ exp(-z) + exp(z);
end