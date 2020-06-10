import numpy as np ## You may NOT modify this line, or your assignment may be graded as 0 point.
## Feel free to import other handy built-in modules here

## self-defined constants
err_threshpold = 1e-5

## Self-defined exceptions
## Please **DO NOT MODIFITY** the following exceptions!
class dtypeError(Exception):
	pass

class SizeUnmatchedError(Exception):
	pass
## Self-defined exceptions end here

## Definition for class KF
class KF:
	def predict(self, x_pre, P_pre, u_k, F, B, Q):
		'''
		Task 1.1: Prediction, NumPy Version
		Input:
			x_pre: n*1 NumPy array
			P_pre: n*n NumPy array
			u_k: m*1 NumPy array
			F: n*n NumPy array
			B: n*m NumPy array
			Q: n*n NumPy array
		Output: 
			x_predicted: n*1 NumPy array
			P_predicted: n*n NumPy array
		'''

		'''Error Check'''
		if (x_pre.dtype != np.float64) or (P_pre.dtype != np.float64) or (u_k.dtype != np.float64) or (F.dtype != np.float64) or (B.dtype != np.float64) or (Q.dtype != np.float64):
			raise dtypeError

		shape_x=x_pre.shape
		shape_P=P_pre.shape
		shape_u=u_k.shape
		shape_F=F.shape
		shape_B=B.shape
		shape_Q=Q.shape
		n=shape_B[0]
		m=shape_B[1]

		if (shape_x != (n,1)) or (shape_P != (n,n)) or (shape_u != (m,1)) or (shape_F != (n,n)) or (shape_Q != (n,n)):
			raise SizeUnmatchedError

		'''Calculation'''
		#step1
		x_predicted=np.matmul(F,x_pre)+np.matmul(B,u_k)
		P_predicted=np.matmul(F,P_pre)

		#step2
		F_transpose = np.transpose(F)
		P_predicted=np.matmul(P_predicted,F_transpose)+Q

		'''Check dtype'''
		if x_predicted.dtype != np.float64:
			x_predicted.astype(np.float64)
		if P_predicted.dtype != np.float64:
			P_predicted.astype(np.float64)

		return x_predicted, P_predicted

	def update(self, x_pre, P_pre, z_k, H, R):
		'''
		Task 1.2: Update, NumPy Version
		Input:
			x_pre: n*1 NumPy array
			P_pre: n*n NumPy array
			z_k: k*1 NumPy array
			H: k*n NumPy array
			R: k*k NumPy array
		Output: 
			x_updated: n*1 NumPy array
			P_updated: n*n NumPy array
		'''
		'''Error Check'''
		if (x_pre.dtype != np.float64) or (P_pre.dtype != np.float64) or (z_k.dtype != np.float64) or (H.dtype != np.float64) or (R.dtype != np.float64):
			raise dtypeError

		shape_x=x_pre.shape
		shape_P=P_pre.shape
		shape_z=z_k.shape
		shape_H=H.shape
		shape_R=R.shape
		k=shape_H[0]
		n=shape_H[1]

		if (shape_x != (n,1)) or (shape_P != (n,n)) or (shape_z != (k,1)) or (shape_R != (k,k)):
			raise SizeUnmatchedError

		'''Calculation'''
		#Step1
		H_transpose=np.transpose(H)
		inv_element=np.matmul(H,P_pre)
		inv_element=np.matmul(inv_element,H_transpose)+R
		inv_element=np.linalg.inv(inv_element)

		K=np.matmul(P_pre,H_transpose)
		K=np.matmul(K,inv_element)

		#Step2
		x_updated=x_pre+np.matmul(K,(z_k-np.matmul(H,x_pre)))

		#Step3
		P_updated=np.matmul(K,H)
		P_updated=P_pre-np.matmul(P_updated,P_pre)

		'''Check dtype'''
		if x_updated.dtype != np.float64:
			x_predicted.astype(np.float64)
		if P_updated.dtype != np.float64:
			P_predicted.astype(np.float64)

		return x_updated, P_updated

if __name__ == "__main__":
	x = [0,1,2]
	P = [[1,2,3],[4,5,6],[7,8,9]]
	x_np = np.array(x, dtype = np.float64).reshape(3, 1)
	P_np = np.array(P, dtype = np.float64)

	u = [0.1, 0.2, 0.3]
	u_np = np.array(u, dtype = np.float64).reshape(3, 1)

	F_np = np.eye(3, dtype = np.float64)
	B_np = np.diag((4,5,6))
	B_np = np.array(B_np, dtype = np.float64)
	Q_np = 0.02 * np.eye(3, dtype = np.float64)

	z_k = [1, 3, 4]
	z_k_np = np.array(z_k, dtype = np.float64).reshape(3, 1)
	H_np = 2 * np.eye(3, dtype = np.float64)
	R_np = 0.03 * np.eye(3, dtype = np.float64)

	# KF implementation testing
	try:
		kf = KF()
		x_predicted, P_predicted = kf.predict(x_np, P_np, u_np, F_np, B_np, Q_np)
		x_updated, P_updated = kf.update(x_predicted, P_predicted, z_k_np, H_np, R_np)
	except:
		print('Implement Incorrect.')
		exit(0)

	x_predicted_ans = np.array([0.4, 2.0, 3.8], dtype = np.float64).reshape(3, 1)
	P_predicted_ans = np.array(
		[[1.02, 2., 3.],
		[4., 5.02, 6.],
		[7., 8., 9.02]], dtype = np.float64)
	x_updated_ans = np.array(
		[[0.53589339],
 		[1.43716531],
 		[2.02934631]], dtype = np.float64)
	P_updated_ans = np.array(
		[[0.00719595, 0.00069144, -0.00035852],
		[ 0.00068505, 0.00613639,  0.00067865],
		[-0.00037131, 0.00067225,  0.00717036]], dtype = np.float64)

	diff_1 = x_predicted - x_predicted_ans
	diff_2 = P_predicted - P_predicted_ans
	diff_3 = x_updated - x_updated_ans
	diff_4 = P_updated - P_updated_ans

	if (diff_1 < err_threshpold).all():
		print('x_predict correct.')
	else:
		print('x_predict wrong.')

	if (diff_2 < err_threshpold).all():
		print('P_predict correct.')
	else:
		print('P_predict wrong.')

	if (diff_3 < err_threshpold).all():
		print('x_updated correct.')
	else:
		print('x_updated wrong.')
	
	if (diff_4 < err_threshpold).all():
		print('P_updated correct.')
	else:
		print('P_updated wrong.')

	## Test exception handling
	x2 = [0,1,2,4]
	x2_np = np.array(x2, dtype = np.float64).reshape(4, 1)

	try:
		x_predicted, P_predicted = kf.predict(x2_np, P_np, u_np, F_np, B_np, Q_np)
	except SizeUnmatchedError:
		print('Pass SizeUnmatchedError test')
	except:
		print('Fail SizeUnmatchedError test')
	else:
		print('Fail SizeUnmatchedError test')

	x_np_2 = np.array(x, dtype = np.float32).reshape(3, 1)
	try:
		x_predicted, P_predicted = kf.predict(x_np_2, P_np, u_np, F_np, B_np, Q_np)
	except dtypeError:
		print('Pass dtypeError test')
	except:
		print('Fail dtypeError test')
	else:
		print('Fail dtypeError test')