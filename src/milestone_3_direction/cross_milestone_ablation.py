"""M3.9 leakage-safe matched-sample cross-milestone ablation."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
ROOT=Path(__file__).resolve().parents[2]; SRC=ROOT/'src'; sys.path.insert(0,str(SRC)) if str(SRC) not in sys.path else None
from milestone_1_regime_detection import walk_forward_validation as m1
from milestone_2_anomaly_detection import walk_forward_validation as m2
EXPORT_YEARS=range(2021,2027); LABELS=['DOWN','FLAT','UP']
BASE=['Return_1D','Return_5D','Volatility_5D','Volatility_20D','Volatility_60D','MA_Distance_20D','MA_Slope_20D','Drawdown_60D']; M1=['M1_P_LOW','M1_P_ELEVATED','M1_P_STRESS']; M2=['M2_Baseline_Score','M2_IF_Score','M2_IF_Training_Percentile','M2_Baseline_Anomaly_Flag']; ALL=BASE+M1+M2
OUT=ROOT/'outputs/milestone_3'; CROSS=OUT/'cross_milestone'; VAL=OUT/'validation'
def rf(): return RandomForestClassifier(n_estimators=500,max_depth=6,min_samples_leaf=10,max_features='sqrt',class_weight=None,random_state=42,n_jobs=-1)
def present_ba(y,p):
 r=[(p[y==c]==c).mean() for c in LABELS if (y==c).any()]; return float(np.mean(r)) if r else np.nan
def export():
 x=pd.read_csv(ROOT/m1.INPUT_FILE,parse_dates=['Date']); z=m2.load_data(); a=[];b=[]
 for y in EXPORT_YEARS:
  q=m1.run_fold(x,y)['Predictions']; a.append(q[['Date','Regime_Name','P_LOW_VOLATILITY_TREND','P_ELEVATED_VOLATILITY_TRANSITION','P_HIGH_VOLATILITY_STRESS']].rename(columns={'Regime_Name':'M1_Regime_Code','P_LOW_VOLATILITY_TREND':'M1_P_LOW','P_ELEVATED_VOLATILITY_TRANSITION':'M1_P_ELEVATED','P_HIGH_VOLATILITY_STRESS':'M1_P_STRESS'}))
  _,s=m2.run_fold(z,y); tr=z[z.Date<pd.Timestamp(f'{y}-01-01')]; fit=m2.fit_models(tr); ts=-fit['isolation'].score_samples(tr[m2.FEATURES].to_numpy(float)); b.append(pd.DataFrame({'Date':s.Date,'M2_Baseline_Anomaly':np.where(s.Baseline_Anomaly,'ANOMALOUS','NORMAL'),'M2_Baseline_Score':s.Baseline_Score,'M2_IF_Score':s.IF_Score,'M2_IF_Training_Percentile':[(ts<=v).mean() for v in s.IF_Score]}))
 return pd.concat(a),pd.concat(b)
def metrics(g):
 y=g.Actual.to_numpy();p=g.Predicted.to_numpy(); labs=[c for c in LABELS if (y==c).any()]; pr,re,f1,su=precision_recall_fscore_support(y,p,labels=LABELS,zero_division=0);i=LABELS.index('DOWN');return dict(OOS_Observations=len(g),Accuracy=accuracy_score(y,p),Balanced_Accuracy=present_ba(y,p),Macro_F1=f1_score(y,p,labels=labs,average='macro',zero_division=0),DOWN_Precision=pr[i] if su[i] else np.nan,DOWN_Recall=re[i] if su[i] else np.nan,DOWN_F1=f1[i] if su[i] else np.nan)
def main():
 CROSS.mkdir(parents=True,exist_ok=True);VAL.mkdir(parents=True,exist_ok=True);a,b=export();a.to_csv(CROSS/'m1_oos_features.csv',index=False);b.to_csv(CROSS/'m2_oos_features.csv',index=False)
 d=pd.read_csv(ROOT/'data/processed/usdtry_direction_features.csv',parse_dates=['Date']);a.Date=pd.to_datetime(a.Date);b.Date=pd.to_datetime(b.Date); m=d.merge(a,on='Date',how='left').merge(b,on='Date',how='left');m['M2_Baseline_Anomaly_Flag']=m.M2_Baseline_Anomaly.map({'NORMAL':0,'ANOMALOUS':1}); joint=m.dropna(subset=ALL+['Target_5D_0p5pct']).copy(); start=joint.Date.min(); years=list(range(start.year+1,2027));diag=pd.DataFrame([{'M3_Total_Rows':len(d),'M3_Rows_2021_2026':int(d.Date.dt.year.between(2021,2026).sum()),'M1_Coverage':int(m.M1_P_LOW.notna().sum()),'M2_Coverage':int(m.M2_IF_Score.notna().sum()),'Joint_Coverage':len(joint),'Earliest_Joint_Date':start,'Latest_Joint_Date':joint.Date.max(),'Dates_Missing_M1':int(m.M1_P_LOW.isna().sum()),'Dates_Missing_M2':int(m.M2_IF_Score.isna().sum()),'Duplicate_Dates':int(d.Date.duplicated().sum()+a.Date.duplicated().sum()+b.Date.duplicated().sum()),'Missing_Values':int(m[ALL].isna().sum().sum())}]);diag.to_csv(VAL/'cross_milestone_feature_alignment.csv',index=False);print('CROSS-MILESTONE DATE ALIGNMENT');print(diag.to_string(index=False));print(f'Cross-milestone training starts: {start.date()}');print(f'First valid ablation test year: {years[0]}');print('Ablation years: '+', '.join(map(str,years)))
 models={'Base_RF_Matched':BASE,'Base_RF_Plus_M1':BASE+M1,'Base_RF_Plus_M2':BASE+M2,'Base_RF_Plus_M1_M2':ALL};fold=[];pred=[];pc=[]
 print('MATCHED ABLATION SAMPLE')
 for y in years:
  tr=joint[joint.Date.dt.year<y];te=joint[joint.Date.dt.year==y];print(f'ABLATION FOLD: {y} train={len(tr)} test={len(te)}')
  for name,fs in models.items():
   model=rf().fit(tr[fs],tr.Target_5D_0p5pct);p=model.predict(te[fs]);g=pd.DataFrame({'Date':te.Date,'Test_Year':y,'Model':name,'Actual':te.Target_5D_0p5pct,'Predicted':p});met=metrics(g);fold.append(dict(Test_Year=y,Model=name,Train_Observations=len(tr),Test_Observations=len(te),**met));pred.append(g);pr,re,f1,su=precision_recall_fscore_support(g.Actual,g.Predicted,labels=LABELS,zero_division=0);pc += [dict(Test_Year=y,Model=name,Class=c,Support=int(n),Precision=x,Recall=z if n else np.nan,F1=q if n else np.nan) for c,x,z,q,n in zip(LABELS,pr,re,f1,su)]
 f=pd.DataFrame(fold);p=pd.concat(pred,ignore_index=True);pc=pd.DataFrame(pc);f.to_csv(VAL/'cross_milestone_ablation_fold_metrics.csv',index=False);p.to_csv(VAL/'cross_milestone_ablation_predictions.csv',index=False);pc.to_csv(VAL/'cross_milestone_ablation_per_class.csv',index=False)
 rows=[]
 for name,g in p.groupby('Model',sort=False): rows.append(dict(Model=name,**metrics(g)))
 s=pd.DataFrame(rows);base=s.loc[s.Model=='Base_RF_Matched'].iloc[0];wins=f.pivot(index='Test_Year',columns='Model',values=['Balanced_Accuracy','Macro_F1']);
 for i,row in s.iterrows():
  s.loc[i,'Delta_BA_vs_Base_RF']=row.Balanced_Accuracy-base.Balanced_Accuracy;s.loc[i,'Delta_F1_vs_Base_RF']=row.Macro_F1-base.Macro_F1;s.loc[i,'Delta_DOWN_Recall_vs_Base']=row.DOWN_Recall-base.DOWN_Recall;s.loc[i,'Delta_DOWN_F1_vs_Base']=row.DOWN_F1-base.DOWN_F1;s.loc[i,'Annual_BA_Wins_vs_Base']=int((wins['Balanced_Accuracy'][row.Model]>wins['Balanced_Accuracy']['Base_RF_Matched']).sum());s.loc[i,'Annual_F1_Wins_vs_Base']=int((wins['Macro_F1'][row.Model]>wins['Macro_F1']['Base_RF_Matched']).sum())
 s.to_csv(VAL/'cross_milestone_ablation_summary.csv',index=False);print('POOLED CROSS-MILESTONE ABLATION');print(s.to_string(index=False));print('MEAN ANNUAL METRICS');print(f.groupby('Model')[['Balanced_Accuracy','Macro_F1']].mean().to_string());print('POOLED PER-CLASS METRICS');print(pc.groupby(['Model','Class'])[['Precision','Recall','F1']].mean().to_string());print('ABLATION SELECTION GATES')
 best='Base_RF_Matched';bestba=base.Balanced_Accuracy
 for _,r in s[s.Model!='Base_RF_Matched'].iterrows():
  gates=[r.Balanced_Accuracy>base.Balanced_Accuracy,r.Macro_F1>base.Macro_F1,r.Annual_BA_Wins_vs_Base>=3,r.DOWN_Recall>=base.DOWN_Recall-0.02,(r.Delta_BA_vs_Base_RF>=.005 or r.Delta_F1_vs_Base_RF>=.005 or r.Delta_DOWN_Recall_vs_Base>=.03 or r.Delta_DOWN_F1_vs_Base>=.03)];print(r.Model, ['PASS' if x else 'FAIL' for x in gates]);
  if all(gates) and r.Balanced_Accuracy>bestba:best=r.Model;bestba=r.Balanced_Accuracy
 print(f'Provisional recommended M3 architecture:\n{best}\n\nReason:\nMatched leakage-safe gates determine the provisional choice; calibration/final selection remains pending.')
if __name__=='__main__':main()
