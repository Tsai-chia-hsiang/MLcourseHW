資工四 408410098 蔡嘉祥

Using a linear regression model $XW=y$ to fit a dataset 

$X=\begin{bmatrix}X_0^T\\X_1^T\\...\\ X_n^T \end{bmatrix},y=\begin{bmatrix}y_0\\y_1\\...\\ y_n^T \end{bmatrix}$,

it's weight $W$ has an analytic solution :

$W=(X^TX)^{-1}X^Ty$

if $X^TX$ is singular, using pseudo inverse of $X$ : $X^\dagger$ instead.

Hence, according to the question, 

$\hat{y} = XW_{\text{LIN}}=XX^\dagger y$

The answer is 3.
