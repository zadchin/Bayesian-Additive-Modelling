import blinpy as bp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM, s
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gammy
from gammy.arraymapper import x

## Difference Priors
def fit_difference_priors(xobs, yobs, xfit, ytrue):

    xobs = np.array(xobs).reshape(-1, 1)
    xfit = np.array(xfit).reshape(-1, 1)

    xobs= xobs.reshape(-1)
    xfit= xfit.reshape(-1)
    
    yobs=yobs
    
    ytrue=ytrue
    generate_gam_spec = lambda B, pri_var: [
    {
        'fun': lambda df: bp.utils.interp_matrix(df['x'].values, xfit, sparse=False),
        'name': 'smoothfun',
        'prior': {
            'B': B,
            'mu': np.zeros(B.shape[0]),
            'cov': pri_var
        }
    }
    ]
    n = len(xfit)
    data = pd.DataFrame({'x': xobs, 'y': yobs})
    
    # case 1: just smoothness prior
    D_smooth = bp.utils.diffmat(n, order=2)
    var_smooth = 0.01*np.ones(D_smooth.shape[0])
    gam_spec_smooth = generate_gam_spec(D_smooth, var_smooth)

    # case 2: periodic smoothness prior
    D_periodic = bp.utils.diffmat(n, order=2, periodic=True)
    var_periodic = 0.01*np.ones(D_periodic.shape[0])
    gam_spec_periodic = generate_gam_spec(D_periodic, var_periodic)

    # symmetric prior
    D_symmetric = bp.utils.symmat(n, nsymm=np.where(xfit >= -np.pi/4)[0][0])
    var_symmetric = 0.01*np.ones(D_symmetric.shape[0])

    # case 3: periodic and symmetric priors combined
    D_per_symm = sparse.vstack((D_periodic, D_symmetric))
    var_per_symm = np.concatenate((var_periodic, var_symmetric))
    gam_spec_both = generate_gam_spec(D_per_symm, var_per_symm)

    yfit_smooth = bp.models.GamModel('y', gam_spec_smooth).fit(data, obs_cov=0.1).post_mu
    yfit_periodic = bp.models.GamModel('y', gam_spec_periodic).fit(data, obs_cov=0.1).post_mu
    yfit_both = bp.models.GamModel('y', gam_spec_both).fit(data, obs_cov=0.1).post_mu

    fig = plt.figure(figsize=(9,3))

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    axs[0].plot(xobs, yobs, 'k.', label='Data')
    axs[0].plot(xfit, ytrue, 'b--', label='Truth')
    axs[0].plot(xfit, yfit_smooth, 'r-', label='Fit')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Smooth', fontsize=10)

    axs[1].plot(xobs, yobs, 'k.')
    axs[1].plot(xfit, ytrue, 'b--')
    axs[1].plot(xfit,yfit_periodic , 'r-')
    axs[1].set_xlabel('x')
    axs[1].set_title('Smooth + Periodic', fontsize=10)


    axs[2].plot(xobs, yobs, 'k.')
    axs[2].plot(xfit, ytrue, 'b--')
    axs[2].plot(xfit, yfit_both, 'r-')
    axs[2].set_xlabel('x')
    axs[2].set_title('Smooth + Periodic + Symmetric', fontsize=10)


    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        bbox_transform=fig.transFigure,
        fancybox=True,
        shadow=True,
        ncol=5,
    )

    plt.tight_layout()

    performance = {
    'Model': ['Smooth', 'Smooth + Periodic', 'Smooth + Periodic + Symmetric'],
    'Mean Squared Error': [
        mean_squared_error(ytrue, yfit_smooth),
        mean_squared_error(ytrue, yfit_periodic),
        mean_squared_error(ytrue, yfit_both)
    ],
    'Root Mean Squared Error': [
        mean_squared_error(ytrue, yfit_smooth, squared=False),
        mean_squared_error(ytrue, yfit_periodic, squared=False),
        mean_squared_error(ytrue, yfit_both, squared=False)
    ],
    'Mean Absolute Error': [
        mean_absolute_error(ytrue, yfit_smooth),
        mean_absolute_error(ytrue, yfit_periodic),
        mean_absolute_error(ytrue, yfit_both)
    ],
    'R2 Score': [
        r2_score(ytrue, yfit_smooth),
        r2_score(ytrue, yfit_periodic),
        r2_score(ytrue, yfit_both)
    ]
    }

    performance_df = pd.DataFrame(performance)
    return [performance_df, fig]

## Linear Regression

def fit_linear_regression(xobs, yobs, xfit, ytrue):

    xobs = np.array(xobs).reshape(-1, 1)
    xfit = np.array(xfit).reshape(-1, 1)
    
    # Fit the linear regression model 
    model = LinearRegression()
    model.fit(xobs, yobs)
    
    yfit = model.predict(xfit)
    
    mse = mean_squared_error(ytrue, yfit)
    
    fig = plt.figure(figsize=(8,4))
    plt.plot(xobs, yobs, 'k.')
    plt.plot(xfit, ytrue, 'b--')
    plt.plot(xfit, yfit, 'r-')
    plt.legend(['Data', 'Truth', 'Fit'])

    plt.xlabel('x')
    plt.title('Linear Regression', fontsize=10)
    
    performance = {
    'Model': ['Linear Regression'],
    'Mean Squared Error': [
        mse
    ],
    'Root Mean Squared Error': [
        mean_squared_error(ytrue, yfit, squared=False)
    ],
    'Mean Absolute Error': [
        mean_absolute_error(ytrue, yfit)
    ],
    'R2 Score': [
        r2_score(ytrue, yfit)
    ]
    }
    
    return [pd.DataFrame(performance), fig]

## Random Forest
def fit_rf_regression(xobs, yobs, xfit, ytrue):

    xobs = np.array(xobs).reshape(-1, 1)
    xfit = np.array(xfit).reshape(-1, 1)
    
    # Fit the Random Forest regression model
    rf_model = RandomForestRegressor()
    rf_model.fit(xobs, yobs)
    
    yfit_rf = rf_model.predict(xfit)
    mse_rf = mean_squared_error(ytrue, yfit_rf)
    
    fig = plt.figure(figsize=(8,4))
    plt.plot(xobs, yobs, 'k.')
    plt.plot(xfit, ytrue, 'b--')
    plt.plot(xfit, yfit_rf, 'r-')
    plt.legend(['Data', 'Truth', 'Random Forest'])
    plt.xlabel('x')
    plt.title('Random Forest Regression', fontsize=10)

    performance = {
    'Model': ['Random Forest'],
    'Mean Squared Error': [
        mse_rf
    ],
    'Root Mean Squared Error': [
        mean_squared_error(ytrue, yfit_rf, squared=False)
    ],
    'Mean Absolute Error': [
        mean_absolute_error(ytrue, yfit_rf)
    ],
    'R2 Score': [
        r2_score(ytrue, yfit_rf)
    ]
    }
    
    return [pd.DataFrame(performance), fig]
    
## Frequentist Penalized GAM
def fit_gam_regression(xobs, yobs, xfit, ytrue):

    xobs = np.array(xobs).reshape(-1, 1)
    xfit = np.array(xfit).reshape(-1, 1)

    # Fit the Frequentist Penalized GAM
    gam_model = LinearGAM(s(0))
    gam_model.fit(xobs, yobs)
    
    yfit_gam = gam_model.predict(xfit)
    mse_gam = mean_squared_error(ytrue, yfit_gam)
    
    # Plot the data, truth, and fit
    fig = plt.figure(figsize=(8,4))
    plt.plot(xobs, yobs, 'k.')
    plt.plot(xfit, ytrue, 'b--')
    plt.plot(xfit, yfit_gam, 'r-')
    plt.legend(['Data', 'Truth', 'AM'])
    plt.xlabel('x')
    plt.title('Frequentist AM', fontsize=10)

    performance = {
    'Model': ['Frequentist AM'],
    'Mean Squared Error': [
        mse_gam
    ],
    'Root Mean Squared Error': [
        mean_squared_error(ytrue, yfit_gam, squared=False)
    ],
    'Mean Absolute Error': [
        mean_absolute_error(ytrue, yfit_gam)
    ],
    'R2 Score': [
        r2_score(ytrue, yfit_gam)
    ]
    }
    
    return [pd.DataFrame(performance), fig]

## Gaussian Process Regression for function fittings
def fit_gp_gammy(xobs, yobs, xfit, ytrue):

    xobs = np.array(xobs).reshape(-1, 1)
    xfit = np.array(xfit).reshape(-1, 1)

    xobs= xobs.reshape(-1)
    xfit= xfit.reshape(-1)

    c = gammy.Scalar()

    corrlen_values = np.linspace(0.1, 2, 10)
    sigma_values = np.linspace(0.1, 2, 10)
    energy_values = np.linspace(0.1, 2, 10)

    best_mse = float('inf')
    best_params = (0, 0, 0)

    ## Hyperparameter tuning
    for corrlen in corrlen_values:
        for sigma in sigma_values:
            for energy in energy_values:
                f = gammy.ExpSquared1d(grid=xfit, corrlen=corrlen, sigma=sigma, energy=energy)
                c = gammy.Scalar()
                formula = f(x) + c
                model = gammy.models.bayespy.GAM(formula).fit(xobs, yobs)
                
                mse = mean_squared_error(ytrue, model.predict(xfit))
                
                if mse < best_mse:
                    best_mse = mse
                    best_params = (corrlen, sigma, energy)

    corrlen = best_params[0]
    sigma = best_params[1]
    energy = best_params[2]

    print(f"Best parameters: corrlen={best_params[0]}, sigma={best_params[1]}, energy={best_params[2]}")
    
    grid = xfit
    exp_squared_model = gammy.models.bayespy.GAM(
    gammy.ExpSquared1d(
        grid=grid,
        corrlen=corrlen,
        sigma=sigma,
        energy=energy
    )(x) + c
    ).fit(xobs, yobs)

    rat_quad_model = gammy.models.bayespy.GAM(
        gammy.RationalQuadratic1d(
            grid=grid,
            corrlen=corrlen,
            alpha=1,
            sigma=sigma,
            energy=energy
        )(x)
    ).fit(xobs, yobs)

    orn_uhl_model = gammy.models.bayespy.GAM(
        gammy.OrnsteinUhlenbeck1d(
            grid=grid,
            corrlen=corrlen,
            sigma=sigma,
            energy=energy
        )(x)
    ).fit(xobs, yobs)

    exp_square_result = exp_squared_model.predict(xfit)
    rat_quad_result = rat_quad_model.predict(xfit)
    orn_uhl_result = orn_uhl_model.predict(xfit)

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    axs[0].plot(xobs, yobs, 'k.', label='Data')
    axs[0].plot(xfit, ytrue, 'b--', label='Truth')
    axs[0].plot(xfit, exp_square_result, 'r-', label='Fit')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Squared Exponential', fontsize=10)

    axs[1].plot(xobs, yobs, 'k.')
    axs[1].plot(xfit, ytrue, 'b--')
    axs[1].plot(xfit, rat_quad_result, 'r-')
    axs[1].set_xlabel('x')
    axs[1].set_title('Rational Quadratic', fontsize=10)

    axs[2].plot(xobs, yobs, 'k.')
    axs[2].plot(xfit, ytrue, 'b--')
    axs[2].plot(xfit, orn_uhl_result, 'r-')
    axs[2].set_xlabel('x')
    axs[2].set_title('Ornstein-Uhlenbeck', fontsize=10)

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        bbox_transform=fig.transFigure,
        fancybox=True,
        shadow=True,
        ncol=5,
    )

    plt.tight_layout()

    performance = {
    'Model': ['exp square', 'rat quad', 'orn uhl'],
    'Mean Squared Error': [
        mean_squared_error(ytrue, exp_square_result),
        mean_squared_error(ytrue, rat_quad_result),
        mean_squared_error(ytrue, orn_uhl_result)
    ],
    'Root Mean Squared Error': [
        mean_squared_error(ytrue, exp_square_result, squared=False),
        mean_squared_error(ytrue, rat_quad_result, squared=False),
        mean_squared_error(ytrue, orn_uhl_result, squared=False)
    ],
    'Mean Absolute Error': [
        mean_absolute_error(ytrue, exp_square_result),
        mean_absolute_error(ytrue, rat_quad_result),
        mean_absolute_error(ytrue, orn_uhl_result)
    ],
    'R2 Score': [
        r2_score(ytrue, exp_square_result),
        r2_score(ytrue, rat_quad_result),
        r2_score(ytrue, orn_uhl_result)
    ]
    }

    performance_df = pd.DataFrame(performance)
    return [performance_df, fig]

## Gaussian Process Regression for data without hyperparameter tuning
def fit_gp_gammy_univariate(X,y):

    xobs = X
    xfit = X
    yobs=y 
    ytrue=y
    grid = np.arange(0, 60, 0.1)
    corrlen = 10
    sigma = 700

    c = gammy.Scalar()

    # Define and fit models with different kernels 
    exp_squared_model = gammy.models.bayespy.GAM(
        gammy.ExpSquared1d(
            grid=grid,
            corrlen=corrlen,
            sigma=sigma,
            energy=0.9
        )(x) + c
    ).fit(X, y)
    rat_quad_model = gammy.models.bayespy.GAM(
        gammy.RationalQuadratic1d(
            grid=grid,
            corrlen=corrlen,
            alpha=1,
            sigma=sigma,
            energy=0.9
        )(x) + c
    ).fit(X, y)
    orn_uhl_model = gammy.models.bayespy.GAM(
        gammy.OrnsteinUhlenbeck1d(
            grid=grid,
            corrlen=corrlen,
            sigma=sigma,
            energy=0.9
        )(x) + c
    ).fit(X, y)

    exp_square_result = exp_squared_model.predict(X)
    rat_quad_result = rat_quad_model.predict(X)
    orn_uhl_result = orn_uhl_model.predict(X)

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    axs[0].plot(xobs, yobs, 'k.', label='Data')
    axs[0].plot(xfit, ytrue, 'b--', label='Truth')
    axs[0].plot(xfit, exp_square_result, 'r-', label='Fit')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Squared Exponential', fontsize=10)

    axs[1].plot(xobs, yobs, 'k.')
    axs[1].plot(xfit, ytrue, 'b--')
    axs[1].plot(xfit, rat_quad_result, 'r-')
    axs[1].set_xlabel('x')
    axs[1].set_title('Rational Quadratic', fontsize=10)

    axs[2].plot(xobs, yobs, 'k.')
    axs[2].plot(xfit, ytrue, 'b--')
    axs[2].plot(xfit, orn_uhl_result, 'r-')
    axs[2].set_xlabel('x')
    axs[2].set_title('Ornstein-Uhlenbeck', fontsize=10)

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        bbox_transform=fig.transFigure,
        fancybox=True,
        shadow=True,
        ncol=5,
    )

    plt.tight_layout()

    performance = {
    'Model': ['exp square', 'rat quad', 'orn uhl'],
    'Mean Squared Error': [
        mean_squared_error(y, exp_square_result),
        mean_squared_error(y, rat_quad_result),
        mean_squared_error(y, orn_uhl_result)
    ],
    'Root Mean Squared Error': [
        mean_squared_error(y, exp_square_result, squared=False),
        mean_squared_error(y, rat_quad_result, squared=False),
        mean_squared_error(y, orn_uhl_result, squared=False)
    ],
    'Mean Absolute Error': [
        mean_absolute_error(y, exp_square_result),
        mean_absolute_error(y, rat_quad_result),
        mean_absolute_error(y, orn_uhl_result)
    ],
    'R2 Score': [
        r2_score(y, exp_square_result),
        r2_score(y, rat_quad_result),
        r2_score(y, orn_uhl_result)
    ]
    }

    performance_df = pd.DataFrame(performance)
    return [performance_df, fig]


    

