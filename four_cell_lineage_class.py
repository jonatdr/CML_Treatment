import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import ParameterGrid
from scipy.integrate import odeint
from matplotlib import pyplot as plt

class Four_cell_lineage:
    def __init__(self):
        pass

    def Hematopoiesis_time_series(self, y, t, k):
        ### ODE equations for Normal Hematopoiesis (Used to obtain Steady States)
        (p0max, p1max, q1max, ## Probabilities
        eta1max, eta2max,     ## Division Rates
        gam1, gam2, gam3, gam4, gam5, ## Feedback Gains
        dL, dM) = k                   ## Death Rates

        dydt = np.zeros(4)

        #FBKS
        p0 = p0max/(1+gam1*y[1])
        eta1 = eta1max/(1+gam2*y[0])
        p1 = p1max/(1+gam3*y[3])
        q1 = q1max/(1+gam4*y[3])
        eta2 = eta2max/(1+gam5*y[0])

        dydt[0] = (2*p0-1)*eta1*y[0]
        dydt[1] = 2*(1-p0)*eta1*y[0] + (2*p1-1)*eta2*y[1]
        dydt[2] =  2*q1*eta2*y[1]-dL*y[2]
        dydt[3] =  2*(1-p1-q1)*eta2*y[1] - dM*y[3]

        return dydt
    def Hematopoiesis_transplant_experiment(self, y, t, k):
        '''HSC, MPP Transplant in Reynaud et al.
           Note: The difference lies in having two identical parallel lineages
           Lineage 1: Normal Development
           Lineage 2: Transplanted cells
        '''
        (p0max, p1max, q1max, ## Probabilities
        eta1max, eta2max,     ## Division Rates
        gam1, gam2, gam3, gam4, gam5, ## Feedback Gains
        dL, dM) = k                   ## Death Rates

        dydt = np.zeros(8)

        #FBKS
        p0 = p0max/(1+gam1*(y[1]+y[5]))
        eta1 = eta1max/(1+gam2*(y[0]+y[4]))
        p1 = p1max/(1+gam3*(y[3]+y[7]))
        q1 = q1max/(1+gam4*(y[3]+y[7]))
        eta2 = eta2max/(1+gam5*(y[0]+y[4]))

        dydt[0] = (2*p0-1)*eta1*y[0]
        dydt[1] = 2*(1-p0)*eta1*y[0] + (2*p1-1)*eta2*y[1]
        dydt[2] =  2*q1*eta2*y[1]-dL*y[2]
        dydt[3] =  2*(1-p1-q1)*eta2*y[1] - dM*y[3]

        dydt[4] = (2*p0-1)*eta1*y[4]
        dydt[5] = 2*(1-p0)*eta1*y[4] + (2*p1-1)*eta2*y[5]
        dydt[6] =  2*q1*eta2*y[5]-dL*y[6]
        dydt[7] =  2*(1-p1-q1)*eta2*y[5] - dM*y[7]

        return dydt

    def Hematopoiesis_TKI_therapy(self, y, t, k):
        ''' TKI therapy
            Two parallel lineages:
            1. Normal Development
            2. Leukemic Development
            Differences between lineages:
            1. Feedback strengths are different,
                a.gam_i for normal development
                b.gam_i_L for leukemic development
            2. Decay term associated to dividing leukemic cells
                a. delta_HSC
                b. delta_MPP
        '''

        (p0max,p1max,q1max,eta1max,eta2max,
         gam1,gam2,gam3,gam4,gam5,
         dL,dM,
         gam1_L,gam2_L,gam3_L,gam4_L,gam5_L,
         delta_HSC,delta_MPP,eta1max_L,eta2max_L,eta1therapy,eta1maxtherapy) = k

        dydt = np.zeros(8)

        #FBKS Normal
        p0 = p0max/(1+gam1*(y[1]+y[5])+eta1therapy)
        eta1 = eta1maxtherapy*eta1max/(1+gam2*(y[0]+y[4]))
        p1 = p1max/(1+gam3*(y[3]+y[7]))
        q1 = q1max/(1+gam4*(y[3]+y[7]))
        eta2 = eta2max/(1+gam5*(y[0]+y[4]))

        #FBKS Leukemic
        p0_L = p0max/(1+gam1_L*gam1*(y[1]+y[5])+eta1therapy)
        eta1_L = eta1maxtherapy*eta1max_L*eta1max/(1+gam2_L*gam2*(y[0]+y[4]))
        p1_L = p1max/(1+gam3_L*gam3*(y[3]+y[7]))
        q1_L = q1max/(1+gam4_L*gam4*(y[3]+y[7]))
        eta2_L = eta2max_L*eta2max/(1+gam5_L*gam5*(y[0]+y[4]))

        #Normal Lineage
        dydt[0] = (2*p0-1)*eta1*y[0]
        dydt[1] = 2*(1-p0)*eta1*y[0] + (2*p1-1)*eta2*y[1]
        dydt[2] =  2*q1*eta2*y[1]-dL*y[2]
        dydt[3] =  2*(1-p1-q1)*eta2*y[1] - dM*y[3]
        #Luekemic Lineage
        dydt[4] = (2*p0_L-1)*eta1_L*y[4] - delta_HSC*eta1_L*y[4]
        dydt[5] = 2*(1-p0_L)*eta1_L*y[4] + (2*p1_L-1)*eta2_L*y[5] - delta_MPP*eta2_L*y[5]
        dydt[6] =  2*q1_L*eta2_L*y[5]-dL*y[6]
        dydt[7] =  2*(1-p1_L-q1_L)*eta2_L*y[5] - dM*y[7]

        return dydt

    def Hematopoiesis_TKI_IFNA(self, y, t, k):
        ''' Combination therapy (TKI + IFN-alpha)

            The difference between this ODE model and previous lies in the
            time dependent HSC division rate (eta1_max) and
            sensitivity to therapy (delta_HSC) '''

        (p0max,p1max,q1max,eta1max,eta2max,
         gam1,gam2,gam3,gam4,gam5,
         dL,dM,
         gam1_L,gam2_L,gam3_L,gam4_L,gam5_L,
         delta_HSC,delta_MPP,n,m) = k

        dydt = np.zeros(8)

        #FBKS
        p0 = p0max/(1+gam1*(y[1]+y[5]))
        eta1 = eta1max/(1+gam2*(y[0]+y[4]))
        p1 = p1max/(1+gam3*(y[3]+y[7]))
        q1 = q1max/(1+gam4*(y[3]+y[7]))
        eta2 = eta2max/(1+gam5*(y[0]+y[4]))

        if n == 0:
            eta1max = eta1max
            delta_HSC = delta_HSC + m*np.abs(np.sin((np.pi/14)*t))
        else:
            eta1max = eta1max + ((t/n)**5)*np.exp(-(2.5*t))
            delta_HSC = delta_HSC + m*np.abs(np.sin((np.pi/14)*t))

        #
        p0_L = p0max/(1+gam1_L*gam1*(y[1]+y[5]))
        eta1_L = eta1max/(1+gam2_L*gam2*(y[0]+y[4]))
        p1_L = p1max/(1+gam3_L*gam3*(y[3]+y[7]))
        q1_L = q1max/(1+gam4_L*gam4*(y[3]+y[7]))
        eta2_L = eta2max/(1+gam5_L*gam5*(y[0]+y[4]))


        dydt[0] = (2*p0-1)*eta1*y[0]
        dydt[1] = 2*(1-p0)*eta1*y[0] + (2*p1-1)*eta2*y[1]
        dydt[2] =  2*q1*eta2*y[1]-dL*y[2]
        dydt[3] =  2*(1-p1-q1)*eta2*y[1] - dM*y[3]

        dydt[4] = (2*p0_L-1)*eta1_L*y[4] - delta_HSC*eta1_L*y[4]
        dydt[5] = 2*(1-p0_L)*eta1_L*y[4] + (2*p1_L-1)*eta2_L*y[5] - delta_MPP*eta2_L*y[5]
        dydt[6] =  2*q1_L*eta2_L*y[5]-dL*y[6]
        dydt[7] =  2*(1-p1_L-q1_L)*eta2_L*y[5] - dM*y[7]

        return dydt

    def solve_ODE(self, ODE_MODEL, data_parameters, data_t0_y0):
        '''This function solves ode given ODE_MODEL'''
        y = odeint(ODE_MODEL, data_t0_y0['y0'],
                   data_t0_y0['t'], args = (data_parameters,))
        return data_parameters, y

    def Transplant_dynamics(self, pars, rad_trans1_trans_2):
        t_y0 = dict(y0=[0.1,1,1,1], t=np.linspace(0,100000,10000))
        _, y_ss = self.solve_ODE(self.Hematopoiesis_time_series,
                              pars, t_y0)
        t_y0_trans = dict(y0=np.concatenate((np.multiply(y_ss[-1],rad_trans1_trans_2[0]),
                                             [rad_trans1_trans_2[1],rad_trans1_trans_2[2],0,0])),
                                              t=np.linspace(0,35,100))
        _, y = self.solve_ODE(self.Hematopoiesis_transplant_experiment,
                              pars, t_y0_trans)
        return t_y0_trans['t'], y
    
    def Transplant_dynamics2(self, pars, rad_trans1_trans_2):
        t_y0 = dict(y0=[0.1,1,1,1], t=np.linspace(0,100000,10000))
        _, y_ss = self.solve_ODE(self.Hematopoiesis_time_series,
                              pars, t_y0)
        t_y0_trans = dict(y0=np.concatenate((y_ss[-1]*5e-3,y_ss[-1]*5e-3)),
                          t=np.linspace(0,365,1000))
        _, y = self.solve_ODE(self.Hematopoiesis_transplant_experiment,
                              pars, t_y0_trans)
        return t_y0_trans['t'], y

    def Plot_ODEs(self, t, y, system):
        '''system:0 Steady State (4 cell system)
           system:1 Transplant Experiment (8 cell system)
           system:2 TKI Therapy (8 cell system)
           '''
        if system == 0:
            cell_labels = ['HSC','MPP','Lymphoid','Myeloid']
        elif system == 1:
            cell_labels = ['HSC','MPP','Lymphoid','Myeloid',
                           'Transplanted HSC','Transplanted MPP',
                           'Transplanted Lymphoid','Transplanted Myeloid']
        elif system == 2:
            cell_labels = ['HSC','MPP','Lymphoid','Myeloid',
                           'L-HSC','L-MPP','L-Lymphoid','L-Myeloid']

        cell_colors = ['k','b','c','g','--k','--b','--c','--g']
        lenofsystem = y.shape
        f,axes = plt.subplots(ncols=lenofsystem[1]//2, figsize=(16,8))
        for idx in range(lenofsystem[1]//2):
            axes[idx].semilogy(t, y[:,(2*idx)]*1e5,cell_colors[2*idx], label = cell_labels[2*idx])
            axes[idx].semilogy(t, y[:,(2*idx+1)]*1e5,cell_colors[2*idx+1], label = cell_labels[2*idx+1])
            axes[idx].set_ylim([1e-2,1e8])
            axes[idx].set_xlabel('Time (Days)')
        axes[0].set_ylabel('Cell Count')
        f.legend()
        f.tight_layout()

    def Plot_therapy_response(self, t, y, col, lab, lstyle):
        BCR_ABL_BCR_1 = 100*(y[:,6]+y[:,7])/(2*(y[:,2]+y[:,3])+y[:,6]+y[:,7])
        plt.semilogy(t, BCR_ABL_BCR_1, col, label = lab, linestyle=lstyle)
        plt.xlabel('Time after Therapy')
        plt.ylabel('BCR-ABL/BCR')

    def Plot_effective_rates(self, tt, yy, p_n, p_l, p_t,
        end_growth, lab, pop, color, opac):

        min = [10] * 10
        max = [0] * 10
        count=0
        f, axes = plt.subplots(nrows=5, ncols=2, figsize = (16,40))
        for t, y in zip(tt, yy):
            ## HSC division
            #axes[0,0].axhline(p_n[count][3]/(1+p_n[count][6]*(y[0,0]+y[0,4])), color='k',linestyle='--', alpha=0.2)
            axes[0,0].plot(t[:1000]-end_growth[count], p_l[count][3]/(1+p_l[count][6]*(y[:1000,0]+y[:1000,4])), color=color[count],linestyle=lab[count], label=pop[count], linewidth=5, alpha = opac[count])
            axes[0,0].plot(t[1000:]-end_growth[count], p_t[count][3]/(1+p_t[count][6]*(y[1000:,0]+y[1000:,4])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[0,0].axvline(x=0,c='black', alpha=0.2)
            axes[0,0].set_xlabel('Time')
            axes[0,0].set_title('Effective HSC division rate $\eta_1$')
            min[0] = np.amin([min[0], np.amin([np.amin(p_l[count][3]/(1+p_l[count][6]*(y[:1000,0]+y[:1000,4]))), np.amin(p_t[count][3]/(1+p_t[count][6]*(y[1000:,0]+y[1000:,4])))])])
            max[0] = np.amax([max[0], np.amax([np.amin(p_l[count][3]/(1+p_l[count][6]*(y[:1000,0]+y[:1000,4]))), np.amax(p_t[count][3]/(1+p_t[count][6]*(y[1000:,0]+y[1000:,4])))])])
            axes[0,0].set_ylim([0.9*min[0],1.1*max[0]])
            
            ## HSC division leukemic
            #axes[0,1].axhline(p_n[count][3]/(1+p_n[count][6]*(y[0,0]+y[0,4])), color='k',linestyle='--', alpha=0.2)
            axes[0,1].plot(t[:1000]-end_growth[count], p_l[count][3]/(1+p_l[count][6]*(y[:1000,0]+y[:1000,4])), color=color[count],linestyle=lab[count], label=pop[count],linewidth=5, alpha = opac[count])
            axes[0,1].plot(t[1000:]-end_growth[count], p_t[count][19]*p_t[count][3]/(1+p_t[count][6]*(y[1000:,0]+y[1000:,4])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[0,1].axvline(x=0,c='black', alpha=0.2)
            axes[0,1].set_xlabel('Time')
            axes[0,1].set_title('Effective HSC division rate $\eta_{1\\rm, L}$')
            min[1] = np.amin([min[1], np.amin([np.amin(p_l[count][3]/(1+p_l[count][6]*(y[:1000,0]+y[:1000,4]))), np.amin(p_t[count][19]*p_t[count][3]/(1+p_t[count][6]*(y[1000:,0]+y[1000:,4])))])])
            max[1] = np.amax([max[1], np.amax([np.amin(p_l[count][3]/(1+p_l[count][6]*(y[:1000,0]+y[:1000,4]))), np.amax(p_t[count][19]*p_t[count][3]/(1+p_t[count][6]*(y[1000:,0]+y[1000:,4])))])])
            axes[0,0].set_ylim([0.9*min[1],1.1*max[1]])
            
            ## MPP division
            #axes[1,0].axhline(p_n[count][4]/(1+p_n[count][9]*(y[0,0]+y[0,4])), color='k',linestyle='--', alpha=0.2)
            axes[1,0].plot(t[:1000]-end_growth[count], p_l[count][4]/(1+p_l[count][9]*(y[:1000,0]+y[:1000,4])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[1,0].plot(t[1000:]-end_growth[count], p_t[count][4]/(1+p_t[count][9]*(y[1000:,0]+y[1000:,4])), color=color[count],linestyle=lab[count],linewidth=5)
            axes[1,0].axvline(x=0,c='black', alpha = opac[count])
            min[2] = np.amin([min[2], np.amin([np.amin(p_l[count][4]/(1+p_l[count][9]*(y[:1000,0]+y[:1000,4]))), np.amin(p_t[count][4]/(1+p_t[count][9]*(y[1000:,0]+y[1000:,4])))])])
            max[2] = np.amax([max[2], np.amax([np.amin(p_l[count][4]/(1+p_l[count][9]*(y[:1000,0]+y[:1000,4]))), np.amax(p_t[count][4]/(1+p_t[count][9]*(y[1000:,0]+y[1000:,4])))])])
            axes[1,0].set_ylim([0.9*min[2],1.1*max[2]])
            axes[1,0].set_xlabel('Time')
            axes[1,0].set_title('Effective MPP division rate $\eta_2$')
            
            ## MPP division leukemic
            #axes[1,1].axhline(p_n[count][4]/(1+p_n[count][9]*(y[0,0]+y[0,4])), color='k',linestyle='--', alpha=0.2)
            axes[1,1].plot(t[:1000]-end_growth[count], p_l[count][4]/(1+p_l[count][9]*(y[:1000,0]+y[:1000,4])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[1,1].plot(t[1000:]-end_growth[count], p_t[count][4]/(1+p_t[count][9]*(y[1000:,0]+y[1000:,4])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[1,1].axvline(x=0,c='black',alpha=0.2)
            min[3] = np.amin([min[3], np.amin([np.amin(p_l[count][4]/(1+p_l[count][9]*(y[:1000,0]+y[:1000,4]))), np.amin(p_t[count][4]/(1+p_t[count][9]*(y[1000:,0]+y[1000:,4])))])])
            max[3] = np.amax([max[3], np.amax([np.amin(p_l[count][4]/(1+p_l[count][9]*(y[:1000,0]+y[:1000,4]))), np.amax(p_t[count][4]/(1+p_t[count][9]*(y[1000:,0]+y[1000:,4])))])])
            axes[1,1].set_ylim([0.9*min[3],1.1*max[3]])
            axes[1,1].set_xlabel('Time')
            axes[1,1].set_title('Effective MPP division rate $\eta_{2\\rm, L}$')

            ## p0 effective
            #axes[2,0].axhline(p_n[count][0]/(1+p_n[count][5]*(y[0,1]+y[0,5])), color='k',linestyle='--', alpha=0.2)
            axes[2,0].plot(t[:1000]-end_growth[count], p_l[count][0]/(1+p_l[count][5]*(y[:1000,1]+y[:1000,5])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[2,0].plot(t[1000:]-end_growth[count], p_t[count][0]/(1+p_t[count][12]*p_t[count][5]*(y[1000:,1]+y[1000:,5])), color=color[count], linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[2,0].axvline(x=0,c='black',alpha=0.2)
            axes[2,0].set_xlabel('Time')
            axes[2,0].set_title('Effective $p_0$ (normal)')
            min[4] = np.amin([min[4], np.amin([np.amin(p_l[count][0]/(1+p_l[count][5]*(y[:1000,1]+y[:1000,5]))), np.amin(p_t[count][0]/(1+p_t[count][12]*p_t[count][5]*(y[1000:,1]+y[1000:,5])))])])
            max[4] = np.amax([max[4], np.amax([np.amin(p_l[count][0]/(1+p_l[count][5]*(y[:1000,1]+y[:1000,5]))), np.amax(p_t[count][0]/(1+p_t[count][12]*p_t[count][5]*(y[1000:,1]+y[1000:,5])))])])
            axes[2,0].set_ylim([0.9*min[4],1.1*max[4]])
                                
            ## p0 effective leukemic
            #axes[2,1].axhline(p_l[count][0]/(1+p_l[count][12]*p_l[count][5]*(y[0,1]+y[0,5])), color='k',linestyle='--', alpha=0.2)
            axes[2,1].plot(t[:1000]-end_growth[count], p_l[count][0]/(1+p_l[count][12]*p_l[count][5]*(y[:1000,1]+y[:1000,5])), color=color[count], linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[2,1].plot(t[1000:]-end_growth[count], p_t[count][0]/(1+p_t[count][12]*p_l[count][5]*(y[1000:,1]+y[1000:,5])), color=color[count], linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[2,1].axvline(x=0,c='black',alpha=0.2)
            axes[2,1].set_xlabel('Time')
            axes[2,1].set_title('Effective $p_{0\\rm, L}$')
            min[5] = np.amin([min[5], np.amin([np.amin(p_l[count][0]/(1+p_l[count][12]*(y[:1000,1]+y[:1000,5]))), np.amin(p_t[count][0]/(1+p_t[count][12]*(y[1000:,1]+y[1000:,5])))])])
            max[5] = np.amax([max[5], np.amax([np.amin(p_l[count][0]/(1+p_l[count][12]*(y[:1000,1]+y[:1000,5]))), np.amax(p_t[count][0]/(1+p_t[count][12]*(y[1000:,1]+y[1000:,5])))])])
            axes[2,1].set_ylim([0.9*min[5],1.1*max[5]])

            #p1 effective
            #axes[3,0].axhline(p_n[count][1]/(1+p_n[count][7]*(y[0,3]+y[0,7])), color='k',linestyle='--', alpha=0.2)
            axes[3,0].plot(t[:1000]-end_growth[count], p_l[count][1]/(1+p_l[count][7]*(y[:1000,3]+y[:1000,7])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[3,0].plot(t[1000:]-end_growth[count], p_t[count][1]/(1+p_t[count][7]*(y[1000:,3]+y[1000:,7])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[3,0].axvline(x=0,c='black',alpha=0.2)
            axes[3,0].set_xlabel('Time')
            axes[3,0].set_title('Effective $p_1$')
            min[6] = np.amin([min[6], np.amin([np.amin(p_l[count][1]/(1+p_l[count][7]*(y[:1000,3]+y[:1000,7]))), np.amin(p_t[count][1]/(1+p_t[count][7]*(y[1000:,3]+y[1000:,7])))])])
            max[6] = np.amax([max[6], np.amax([np.amin(p_l[count][1]/(1+p_l[count][7]*(y[:1000,3]+y[:1000,7]))), np.amax(p_t[count][1]/(1+p_t[count][7]*(y[1000:,3]+y[1000:,7])))])])
            axes[3,0].set_ylim([0.9*min[6],1.1*max[6]])
            
            #p1 effective leukemic
            #axes[3,1].axhline(p_n[count][1]/(1+p_n[count][7]*(y[0,3]+y[0,7])), color='k',linestyle='--', alpha=0.2)
            axes[3,1].plot(t[:1000]-end_growth[count], p_l[count][1]/(1+p_l[count][7]*(y[:1000,3]+y[:1000,7])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[3,1].plot(t[1000:]-end_growth[count], p_t[count][1]/(1+p_t[count][7]*(y[1000:,3]+y[1000:,7])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[3,1].axvline(x=0,c='black',alpha=0.2)
            axes[3,1].set_xlabel('Time')
            axes[3,1].set_title('Effective $p_{1\\rm, L}$')
            min[7] = np.amin([min[7], np.amin([np.amin(p_l[count][1]/(1+p_l[count][7]*(y[:1000,3]+y[:1000,7]))), np.amin(p_t[count][1]/(1+p_t[count][7]*(y[1000:,3]+y[1000:,7])))])])
            max[7] = np.amax([max[7], np.amax([np.amin(p_l[count][1]/(1+p_l[count][7]*(y[:1000,3]+y[:1000,7]))), np.amax(p_t[count][1]/(1+p_t[count][7]*(y[1000:,3]+y[1000:,7])))])])
            axes[3,1].set_ylim([0.9*min[7],1.1*max[7]])
                                
            #q1 effective
            #axes[4,0].axhline(p_n[count][2]/(1+p_n[count][8]*(y[0,3]+y[0,7])), color='k',linestyle='--', alpha=0.2)
            axes[4,0].plot(t[:1000]-end_growth[count], p_l[count][2]/(1+p_l[count][8]*(y[:1000,3]+y[:1000,7])), color=color[count], linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[4,0].plot(t[1000:]-end_growth[count], p_t[count][2]/(1+p_t[count][8]*(y[1000:,3]+y[1000:,7])), color=color[count], linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[4,0].axvline(x=0,c='black',alpha=0.2)
            axes[4,0].set_xlabel('Time')
            axes[4,0].set_title('Effective $q_1$')
            min[8] = np.amin([min[8], np.amin([np.amin(p_l[count][2]/(1+p_l[count][8]*(y[:1000,3]+y[:1000,7]))), np.amin(p_t[count][2]/(1+p_t[count][8]*(y[1000:,3]+y[1000:,7])))])])
            max[8] = np.amax([max[8], np.amax([np.amin(p_l[count][2]/(1+p_l[count][8]*(y[:1000,3]+y[:1000,7]))), np.amax(p_t[count][2]/(1+p_t[count][8]*(y[1000:,3]+y[1000:,7])))])])
            axes[4,0].set_ylim([0.9*min[8],1.1*max[8]])
            
            #q1 effective leukemic
            #axes[4,1].axhline(p_n[count][2]/(1+p_n[count][8]*(y[0,3]+y[0,7])), color='k',linestyle='--', alpha=0.2)
            axes[4,1].plot(t[:1000]-end_growth[count], p_l[count][2]/(1+p_l[count][8]*(y[:1000,3]+y[:1000,7])), color=color[count], linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[4,1].plot(t[1000:]-end_growth[count], p_t[count][2]/(1+p_t[count][8]*(y[1000:,3]+y[1000:,7])), color=color[count], linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[4,1].axvline(x=0,c='black',alpha=0.2)
            axes[4,1].set_xlabel('Time')
            axes[4,1].set_title('Effective $q_{1\\rm, L}$')
            min[9] = np.amin([min[9], np.amin([np.amin(p_l[count][2]/(1+p_l[count][8]*(y[:1000,3]+y[:1000,7]))), np.amin(p_t[count][2]/(1+p_t[count][8]*(y[1000:,3]+y[1000:,7])))])])
            max[9] = np.amax([max[9], np.amax([np.amin(p_l[count][2]/(1+p_l[count][8]*(y[:1000,3]+y[:1000,7]))), np.amax(p_t[count][2]/(1+p_t[count][8]*(y[1000:,3]+y[1000:,7])))])])
            axes[4,1].set_ylim([0.9*min[9],1.1*max[9]])

            count+=1
        #axes[0,0].legend(loc = 'best')
        f.tight_layout()
        
    def Plot_effective_rates_separate(self, tt, yy, p_n, p_l, p_t,
        end_growth, lab, pop, color, opac):

        min = [10] * 10
        max = [0] * 10
        count=0
        f, axes = plt.subplots(nrows=5, ncols=2, figsize = (16,40))
        for t, y in zip(tt, yy):
            ## HSC division
            #axes[0,0].axhline(p_n[count][3]/(1+p_n[count][6]*(y[0,0]+y[0,4])), color='k',linestyle='--', alpha=0.2)
            axes[0,0].plot(t-end_growth[count], p_l[count][3]/(1+p_l[count][6]*(y[:,0]+y[:,4])), color=color[count],linestyle=lab[count], label=pop[count], linewidth=5, alpha = opac[count])
            axes[0,0].set_xlabel('Time')
            axes[0,0].set_title('Effective HSC division rate $\eta_1$')
            min[0] = np.amin([min[0], np.amin(p_l[count][3]/(1+p_l[count][6]*(y[:,0]+y[:,4])))])
            max[0] = np.amax([max[0], np.amax(p_l[count][3]/(1+p_l[count][6]*(y[:,0]+y[:,4])))])
            axes[0,0].set_ylim([0.9*min[0],1.1*max[0]])
            
            ## HSC division leukemic
            #axes[0,1].axhline(p_n[count][3]/(1+p_n[count][6]*(y[0,0]+y[0,4])), color='k',linestyle='--', alpha=0.2)
            axes[0,1].plot(t-end_growth[count], p_l[count][3]/(1+p_l[count][6]*(y[:,0]+y[:,4])), color=color[count],linestyle=lab[count], label=pop[count],linewidth=5, alpha = opac[count])
            axes[0,1].set_xlabel('Time')
            axes[0,1].set_title('Effective HSC division rate $\eta_{1\\rm, L}$')
            min[1] = np.amin([min[1], np.amin(p_l[count][3]/(1+p_l[count][6]*(y[:,0]+y[:,4])))])
            max[1] = np.amax([max[1], np.amax(p_l[count][3]/(1+p_l[count][6]*(y[:,0]+y[:,4])))])
            axes[0,0].set_ylim([0.9*min[1],1.1*max[1]])
            
            ## MPP division
            #axes[1,0].axhline(p_n[count][4]/(1+p_n[count][9]*(y[0,0]+y[0,4])), color='k',linestyle='--', alpha=0.2)
            axes[1,0].plot(t-end_growth[count], p_l[count][4]/(1+p_l[count][9]*(y[:,0]+y[:,4])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            min[2] = np.amin([min[2], np.amin(p_l[count][4]/(1+p_l[count][9]*(y[:,0]+y[:,4])))])
            max[2] = np.amax([max[2], np.amax(p_l[count][4]/(1+p_l[count][9]*(y[:,0]+y[:,4])))])
            axes[1,0].set_ylim([0.9*min[2],1.1*max[2]])
            axes[1,0].set_xlabel('Time')
            axes[1,0].set_title('Effective MPP division rate $\eta_2$')
            
            ## MPP division leukemic
            #axes[1,1].axhline(p_n[count][4]/(1+p_n[count][9]*(y[0,0]+y[0,4])), color='k',linestyle='--', alpha=0.2)
            axes[1,1].plot(t-end_growth[count], p_l[count][4]/(1+p_l[count][9]*(y[:,0]+y[:,4])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            min[3] = np.amin([min[3], np.amin(p_l[count][4]/(1+p_l[count][9]*(y[:,0]+y[:,4])))])
            max[3] = np.amax([max[3], np.amax(p_l[count][4]/(1+p_l[count][9]*(y[:,0]+y[:,4])))])
            axes[1,1].set_ylim([0.9*min[3],1.1*max[3]])
            axes[1,1].set_xlabel('Time')
            axes[1,1].set_title('Effective MPP division rate $\eta_{2\\rm, L}$')

            ## p0 effective
            #axes[2,0].axhline(p_n[count][0]/(1+p_n[count][5]*(y[0,1]+y[0,5])), color='k',linestyle='--', alpha=0.2)
            axes[2,0].plot(t-end_growth[count], p_l[count][0]/(1+p_l[count][5]*(y[:,1]+y[:,5])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[2,0].set_xlabel('Time')
            axes[2,0].set_title('Effective $p_0$ (normal)')
            min[4] = np.amin([min[4], np.amin(p_l[count][0]/(1+p_l[count][5]*(y[:,1]+y[:,5])))])
            max[4] = np.amax([max[4], np.amax(p_l[count][0]/(1+p_l[count][5]*(y[:,1]+y[:,5])))])
            axes[2,0].set_ylim([0.9*min[4],1.1*max[4]])
                                
            ## p0 effective leukemic
            #axes[2,1].axhline(p_l[count][0]/(1+p_l[count][12]*p_l[count][5]*(y[0,1]+y[0,5])), color='k',linestyle='--', alpha=0.2)
            axes[2,1].plot(t-end_growth[count], p_l[count][0]/(1+p_l[count][12]*p_l[count][5]*(y[:,1]+y[:,5])), color=color[count], linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[2,1].set_xlabel('Time')
            axes[2,1].set_title('Effective $p_{0\\rm, L}$')
            min[5] = np.amin([min[5], np.amin(p_l[count][0]/(1+p_l[count][12]*(y[:,1]+y[:,5])))])
            max[5] = np.amax([max[5], np.amax(p_l[count][0]/(1+p_l[count][12]*(y[:,1]+y[:,5])))])
            axes[2,1].set_ylim([0.9*min[5],1.1*max[5]])

            #p1 effective
            #axes[3,0].axhline(p_n[count][1]/(1+p_n[count][7]*(y[0,3]+y[0,7])), color='k',linestyle='--', alpha=0.2)
            axes[3,0].plot(t-end_growth[count], p_l[count][1]/(1+p_l[count][7]*(y[:,3]+y[:,7])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[3,0].set_xlabel('Time')
            axes[3,0].set_title('Effective $p_1$')
            min[6] = np.amin([min[6], np.amin(p_l[count][1]/(1+p_l[count][7]*(y[:,3]+y[:,7])))])
            max[6] = np.amax([max[6], np.amax(p_l[count][1]/(1+p_l[count][7]*(y[:,3]+y[:,7])))])
            axes[3,0].set_ylim([0.9*min[6],1.1*max[6]])
            
            #p1 effective leukemic
            #axes[3,1].axhline(p_n[count][1]/(1+p_n[count][7]*(y[0,3]+y[0,7])), color='k',linestyle='--', alpha=0.2)
            axes[3,1].plot(t-end_growth[count], p_l[count][1]/(1+p_l[count][7]*(y[:,3]+y[:,7])), color=color[count],linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[3,1].set_xlabel('Time')
            axes[3,1].set_title('Effective $p_{1\\rm, L}$')
            min[7] = np.amin([min[7], np.amin(p_l[count][1]/(1+p_l[count][7]*(y[:,3]+y[:,7])))])
            max[7] = np.amax([max[7], np.amax(p_l[count][1]/(1+p_l[count][7]*(y[:,3]+y[:,7])))])
            axes[3,1].set_ylim([0.9*min[7],1.1*max[7]])
                                
            #q1 effective
            #axes[4,0].axhline(p_n[count][2]/(1+p_n[count][8]*(y[0,3]+y[0,7])), color='k',linestyle='--', alpha=0.2)
            axes[4,0].plot(t-end_growth[count], p_l[count][2]/(1+p_l[count][8]*(y[:,3]+y[:,7])), color=color[count], linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[4,0].set_xlabel('Time')
            axes[4,0].set_title('Effective $q_1$')
            min[8] = np.amin([min[8], np.amin(p_l[count][2]/(1+p_l[count][8]*(y[:,3]+y[:,7])))])
            max[8] = np.amax([max[8], np.amax(p_l[count][2]/(1+p_l[count][8]*(y[:,3]+y[:,7])))])
            axes[4,0].set_ylim([0.9*min[8],1.1*max[8]])
            
            #q1 effective leukemic
            #axes[4,1].axhline(p_n[count][2]/(1+p_n[count][8]*(y[0,3]+y[0,7])), color='k',linestyle='--', alpha=0.2)
            axes[4,1].plot(t-end_growth[count], p_l[count][2]/(1+p_l[count][8]*(y[:,3]+y[:,7])), color=color[count], linestyle=lab[count],linewidth=5, alpha = opac[count])
            axes[4,1].set_xlabel('Time')
            axes[4,1].set_title('Effective $q_{1\\rm, L}$')
            min[9] = np.amin([min[9], np.amin(p_l[count][2]/(1+p_l[count][8]*(y[:,3]+y[:,7])))])
            max[9] = np.amax([max[9], np.amax(p_l[count][2]/(1+p_l[count][8]*(y[:,3]+y[:,7])))])
            axes[4,1].set_ylim([0.9*min[9],1.1*max[9]])

            count+=1
        #axes[0,0].legend(loc = 'best')
        f.tight_layout()
        

    def Leukemia_evolution_TKI_ODE_dyanamics(self,
    paramset_n, paramset_l, paramset_t,
    end_growth, end_therapy):

        y0_t_n= dict(t=np.linspace(0,100000,100000),
                     y0=[.1,1,10,10,0,0,0,0])
        # Solve to Steady State
        _, y_SS = self.solve_ODE(self.Hematopoiesis_TKI_therapy,
                                 paramset_n, y0_t_n)

        y0_t_l = dict(t=np.linspace(0, end_growth, 1000),
                      y0= np.concatenate((y_SS[-1,:4],[.1, 0.0, 0.0, 0.0])))
        # Leukemia growth
        _, y_l = self.solve_ODE(self.Hematopoiesis_TKI_therapy,
                                paramset_l,
                                y0_t_l)

        y0_t_t = dict(t=np.linspace(0,end_therapy,1000),
                      y0=y_l[-1])
        # Therapy
        theta_t, y_t = self.solve_ODE(self.Hematopoiesis_TKI_therapy,
                                      paramset_t,
                                      y0_t_t)

        t = np.append(y0_t_l['t'], y0_t_t['t'] + y0_t_l['t'][-1], axis=0)
        y = np.append(y_l, y_t, axis=0)

        return theta_t, t, y

    def Leukemia_evolution_TKI_IFNA_ODE_dyanamics(self,
    paramset_n, paramset_l, paramset_t,
    end_growth, weeks_treatment):

        y0_t_n = dict(t=np.linspace(0,100000,100000), y0=[.1,1,10,10,0,0,0,0])

        # Solve to Steady State
        _, y_SS = self.solve_ODE(self.Hematopoiesis_TKI_therapy,
                                 paramset_n, y0_t_n)

        y0_t_l = dict(t=np.linspace(0, end_growth, 1000),
                      y0= np.concatenate((y_SS[-1,:4],[.1, 0.0, 0.0, 0.0])))
        # Leukemia growth
        _, y_l = self.solve_ODE(self.Hematopoiesis_TKI_therapy,
                                paramset_l,
                                y0_t_l)

        y0_t_t = dict(t=np.linspace(0,14,1000), y0=y_l[-1])

        #self.Hematopoiesis_Time_Series, data_t0_y0['y0'], data_t0_y0['t'], args = (theta,)
        y_t = odeint(self.Hematopoiesis_TKI_IFNA, y0_t_t['y0'], y0_t_t['t'], args=(paramset_t,))

        t = np.concatenate((y0_t_l['t'], y0_t_l['t'][-1] +  y0_t_t['t']))
        y_l = np.concatenate((y_l, y_t))

        for i in range(weeks_treatment):
            y0_t_therapy = dict(t=np.linspace(0,14,1000), y0=y_l[-1])
            t = np.append(t, t[-1] + y0_t_therapy['t'], axis=0)
            y_therapy = odeint(self.Hematopoiesis_TKI_IFNA,
                               y0_t_therapy['y0'],
                               y0_t_therapy['t'],
                               args=(paramset_t,))
            y_l = np.append(y_l, y_therapy, axis=0)

        return t, y_l

# df_IFNA = np.array([5.40220268e-01, 4.20039036e-01, 1.05677019e-01, 3.74933433e-01,
#        1.30941498e+00, 8.04405357e-02, 4.45841266e-01, 7.85621003e-05,
#        1.84014128e-05, 1.39464368e-03, 5.81058193e-02, 1.71103507e-01])

# fcl_therapy = Four_cell_lineage();
# #
# # t_y0 = dict(y0=[0.1,1,1,1], t=np.linspace(0,100000,10000))
# # t, y = fcl_therapy.Transplant_dynamics(df_IFNA, [0.25, 0.4,0.0])
# #
# # fcl_therapy.Plot_ODEs(t,y,1)
# #
# #
# gam1_L = 0.5
# TKI_paramset_normal = np.concatenate((df_IFNA, [0,0,0,0,0,0,0]));
# TKI_paramset_leukemia =  np.concatenate((df_IFNA,[gam1_L,1,1,1,1,0,0]));
# TKI_paramset_therapy = np.concatenate((df_IFNA,[gam1_L,1,1,1,1,0.07,20]));
# #
# theta, t, y = fcl_therapy.Leukemia_evolution_TKI_ODE_dyanamics(
# TKI_paramset_normal, TKI_paramset_leukemia, TKI_paramset_therapy, 0.5*365, 1300)
#
# theta, t_l, y_l = fcl_therapy.Leukemia_evolution_TKI_ODE_dyanamics(
# TKI_paramset_normal, TKI_paramset_leukemia, TKI_paramset_therapy, 3*365, 400)
#
# # fcl_therapy.Plot_ODEs(t,y,2)
# #
# # plt.figure()

# # fcl_therapy.Plot_therapy_response(t_l[t_l>=3*365]-3*365, y_l[t_l>=3*365], 'Late', '--')
# #
# fcl_therapy.Plot_effective_rates([t,t_l], [y,y_l],
# TKI_paramset_normal, TKI_paramset_leukemia, TKI_paramset_therapy, [0.5*365, 3*365], ['-','-.'], ['Early','Late'])
# fcl_therapy.Plot_effective_rates(t_l, y_l,
# TKI_paramset_normal, TKI_paramset_leukemia, TKI_paramset_therapy, 3*365, '-.', 'Late')
# #
#
# for j, k in zip([[1,2,3],[4,5,6]],[[3,4,5],[6,7,8]]):
#     plt.plot(j,k)


#
# TKI_paramset_normal = np.concatenate((df_IFNA, [0,0,0,0,0,0,0]));
# TKI_paramset_leukemia =  np.concatenate((df_IFNA,[gam1_L,1,1,1,1,0,0]));
# TKI_paramset_combination_therapy = np.concatenate((df_IFNA,[gam1_L,1,1,1,1,0.07,20,0,0]));
#
# t,y = fcl_therapy.Leukemia_evolution_TKI_IFNA_ODE_dyanamics(
# TKI_paramset_normal, TKI_paramset_leukemia, TKI_paramset_combination_therapy,
# 365*0.5, 40)

#fcl_therapy.Plot_ODEs(t,y,2)
#fcl_therapy.Plot_therapy_response(t[t>=0.5*365]-0.5*365, y[t>=0.5*365], 'k', 'Early', '-')
# plt.figure()
# fcl_therapy.Plot_therapy_response(t[t>=0.5*365]-0.5*365, y[t>=0.5*365], 'Early', '-')
