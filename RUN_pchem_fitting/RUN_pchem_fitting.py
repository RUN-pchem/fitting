import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display,clear_output
from scipy.optimize import curve_fit,least_squares 

def fit():
	wl = widgets.Layout(width='80%',height='24pt')
	wl2 = widgets.Layout(width='80%',height='1.5in')
	wbl = widgets.Layout(width='2in',height='0.25in')
	ws = {'description_width':'initial'}

	out = widgets.Output()

	ydata = widgets.Textarea(value='',placeholder='Enter Dependent Data Here (Y)',description='Y data ',layout=wl2,style=ws)
	xdata = widgets.Textarea(value='',placeholder='Enter Independent Data Here (X)',description='X data ',layout=wl2,style=ws)

	# odata = widgets.Textarea(value='',placeholder='Output Data (Model)',description='Model   ',layout=wl2,style=ws)
	fdata = widgets.Textarea(value='',placeholder='Information about the model',description=  'Fit Info',layout=wl2,style=ws)
	
	# dropdown_fxn = widgets.Dropdown(value='Exponential Decay', options=['Linear','Quadratic','Exponential Decay','Double Exponential'],description='Fitting Function',style=ws)
	dropdown_fxn = widgets.Dropdown(value='Exponential Decay', options=['Linear','Quadratic','Cubic','Exponential Decay',],description='Fitting Function',style=ws)
	button_fit = widgets.Button(description='Fit',layout=wbl,style=ws)

	hbox = widgets.HBox([dropdown_fxn,button_fit,])
	total = widgets.VBox([xdata,ydata,fdata,hbox])

	def show_ui():
		with out:
			out.clear_output()
			display(total)

	def fxn_exp1(x,A1,k1,B):
		return A1 * np.exp(-k1*(x)) + B
	def fxn_exp2(x,A1,k1,A2,k2,B):
		if k1>=k2:
			return x*np.nan
		return A1 * np.exp(-k1*(x)) + A2 * np.exp(-k2*(x)) + B
	def fxn_linear(x,A,B):
		return A*x+B
	def fxn_quadratic(x,A,B,C):
		return A*x**2.+B*x+C
	def fxn_cubic(x,A,B,C,D):
		return A*x**3.+B*x**2.+C*x+D

	def get_fxn():
		if dropdown_fxn.value == 'Linear':
			fxn = fxn_linear
		elif dropdown_fxn.value == 'Quadratic':
			fxn = fxn_quadratic
		elif dropdown_fxn.value == 'Cubic':
			fxn = fxn_cubic
		elif dropdown_fxn.value == 'Exponential Decay':
			fxn = fxn_exp1
		elif dropdown_fxn.value == 'Double Exponential':
			fxn = fxn_exp2
		def residual_fxn(theta,x,y):
			return y - fxn(x,*theta)
		return fxn,residual_fxn

	def make_plot(x,y,theta=None):
		fxn,residual_fxn = get_fxn()
		
		if not theta is None:
			fig,ax = plt.subplots(2,sharex=True,figsize=(4,4),height_ratios=[4, 1],)
			ax[0].plot(x,y,color='black',ls='none',marker='o')
			ax[1].axhline(y=0,color='black')

			fit_x = np.linspace(x.min(),x.max(),1000)
			fit_y = fxn(fit_x,*theta)

			residual = residual_fxn(theta,x,y)

			ax[0].plot(fit_x,fit_y,color='tab:red')
			ax[1].plot(x,residual,color='tab:red')
			delta = np.max(np.abs(residual))*1.05
			ax[1].set_ylim(-delta,delta)
			ax[1].set_ylabel('Residual')
			ax[1].set_xlabel('Independent Data')
			ax[0].set_xlim(x.min(),x.max())
			ax[0].set_ylabel('Dependent Data')
		else:
			fig,ax = plt.subplots(1,sharex=True,figsize=(4,3),)
			ax.plot(x,y,color='black')
			ax.set_xlim(x.min(),x.max())
			ax.set_xlabel('Independent Data')
			ax.set_ylabel('Dependent Data')

		plt.tight_layout()
		return fig,ax

	def parse(ss):
		try:
			ss = ss.lstrip().rstrip()
			if ss.count(',') > 0:
				d = np.array([float(ssi) for ssi in ss.split(',')])
			elif ss.count('\t') > 0:
				d = np.array([float(ssi) for ssi in ss.split('\t')])
			elif ss.count(' ') > 0:
				d = np.array([float(ssi) for ssi in ss.split(' ')])
			elif ss.count('\n') > 1:
				d = np.array([float(ssi) for ssi in ss.split('\n')])
			else:
				raise Exception('ERROR: No delimiter found?')
			return d
		except Exception as e:
			print(e)
			print(f'ERROR: Could not parse the input data')
			
	def load():
		if xdata.value == '' or ydata.value == '':
			return None,None
		x = parse(xdata.value)
		y = parse(ydata.value)
		return x,y

	def guess_theta():
		x,y = load()
		if x is None:
			return None
		
		# B = y.min()
		# k,A = np.polyfit(x,np.log(y-B),1)

		# A = np.exp(A)
		# # A = y.max()-y.min() if y.argmax() < y.argmin() else y.min()-y.max()
		# # k = 10./(x.max()-x.min())
		# # B = y.mean()
		
		if dropdown_fxn.value == 'Linear':
			return np.polyfit(x,y,1)
		elif dropdown_fxn.value == 'Quadratic':
			return np.polyfit(x,y,2)
		elif dropdown_fxn.value == 'Cubic':
			return np.polyfit(x,y,3)
		elif dropdown_fxn.value == 'Exponential Decay':
			k = 1./(x.max()-x.min())*10.
			B = y[-1]
			A = y[0]-y[-1]
			return np.array((A,k,B))
		# elif dropdown_fxn.value == 'Double Exponential':
		# 	return np.array((A/2.,k/2.,A/2.,k*2,B))

	def click_fit(b):
		with out:
			show_ui()

			x,y = load()
			if x is None:
				print('ERROR: Loading Failed')
				return

			theta = guess_theta()
			if 	theta is None:
				print('ERROR: Guessing Failed')
				return
						
			fxn,residual_fxn = get_fxn()
			# result = least_squares(residual_fxn,x0=theta,args=(x,y))
			# theta = result.x
			theta,cov = curve_fit(fxn,x,y,theta,maxfev=100000)

			# if not result.success:
			# 	raise Exception('Failed')

			ss_res = np.sum(residual_fxn(theta,x,y)**2.)
			# cov = ss_res/float(x.size-result.x.size) * np.linalg.inv(np.dot(result.jac.T,result.jac))
			sig = np.sqrt(np.diag(cov))
			ss_tot = np.sum((y - np.mean(y))**2.)
			r_squared = 1.-(ss_res/ss_tot)

			if dropdown_fxn.value == 'Linear':
				params = ['m','b']
			elif dropdown_fxn.value == 'Quadratic':
				params = ['A','B','C']
			elif dropdown_fxn.value == 'Cubic':
				params = ['A','B','C','D']
			elif dropdown_fxn.value == 'Exponential Decay':
				params = ['A','k','B']
			elif dropdown_fxn.value == 'Double Exponential':
				params = ['A1','k1','A2','k2','B']

			fstr = 'Fitting Results: '
			if dropdown_fxn.value == "Linear":
				fstr += 'y = m*x+b\n'
			elif dropdown_fxn.value == "Quadratic":
				fstr += 'y = A*x^2 + B*x + C\n'
			elif dropdown_fxn.value == "Cubic":
				fstr += 'y = A*x^3 + B*x^2 + C*x + D\n'
			elif dropdown_fxn.value == "Exponential Decay":
				fstr += 'y = A*exp[-k*x] + B\n'
			elif dropdown_fxn.value == "Double Exponential":
				fstr += 'y = A1*exp[-k1*x] + A2*exp[-k2*x] + B\n'
			for i in range(len(params)):
				fstr += f'{params[i]} = {theta[i]:.3e} +/- {sig[i]:.3e}\n'
			fstr += f'R^2 = {r_squared:.6f}\n'
			fdata.value = fstr

			fig,ax = make_plot(x,y,theta)

			plt.show()
			plt.close()

	button_fit.on_click(click_fit)
	show_ui()
	display(out)
