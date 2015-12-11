function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); % m is number of training examples here 5000
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


a1=[ones(m,1) X];  % a1 is 5000*401 matrix
                    % Theta1 is 25 * 401 matrix
z2=Theta1*a1';      % z2 is 25*5000 matrix
z2=z2';             % z2 is 5000* 25 matrix

%z22=a1*Theta1';     % z22 is 5000* 25 matrix and z22 is equal to z2
%a22=sigmoid(z22);

a2=sigmoid(z2); 
a2=[ones(m,1) a2];  % a2 is 5000 * 26 matrix

z3=Theta2*a2';      % z3 is 10*5000 vector
z3=z3';             % z3 is 5000*10 vector
a3=sigmoid(z3);     
%size(a3)
y_vector=zeros(m,num_labels);   %y_vector is 5000*10 vector
%size(y_vector)
for i=1:m
    y_vector(i,y(i))=1;
end


J=(1/m)*sum(sum(-y_vector.*log(a3) - (1-y_vector).*log(1-a3)));


reg=sum(sum(Theta1(:,2:(input_layer_size+1)).^2))+sum(sum(Theta2(:,2:(hidden_layer_size+1)).^2));
reg=(lambda/(2*m))*reg;
J=J+reg;


% calculating grad and using back propagation algorithm to set the correct
% parameters.

for t=1:m
    a1=X(t,:)';         % a1 is 400 vector;
    a1=[1;a1];          %Theta is 25*401 matrix
    z2=Theta1 * a1;     %z2 is 25 vector;
    a2=[1;sigmoid(z2)];        %Theta2 is 10 * 26 matrix;
    z3=Theta2*a2;       %a3 is 10 vector
    a3=sigmoid(z3);
    
    y_vec=([1:num_labels]==y(t))';      % y_vec is a 10 vector
    delta3=a3-y_vec;
    
    
    delta2=(Theta2'*delta3).*[1;sigmoidGradient(z2)];
    delta2=delta2(2:end);
    
    %update the capital delta values
    Theta1_grad= Theta1_grad + delta2*a1';
    Theta2_grad=Theta2_grad+delta3*a2';
      
end

Theta1_grad=(1/m)*Theta1_grad;
Theta2_grad=(1/m)*Theta2_grad;


Theta1_grad=Theta1_grad+(lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad=Theta2_grad+(lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)];






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
