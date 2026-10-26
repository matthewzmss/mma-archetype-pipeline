import pandas as pd, json, sys
d=pd.read_csv('out/fighter_archetypes.csv')
A=["One Punch Man","Brawler","Pressure Fighter","Point Fighter","Sniper","Defensive Technician","Submission Threat","Grappler / Wrestler","Point-Fighting Wrestler","Ground-and-Pound Wrestler","All-in-One"]
stats=[('slpm','Sig. landed / min',2,0),('sapm','Sig. absorbed / min',2,1),('sig_acc','Strike accuracy',3,0),('sig_def','Strike defense',3,0),('str_diff','Strike differential',2,0),
('kd15','Knockdowns / 15',2,0),('kd_per_100_landed','KDs per 100 landed',2,0),('kd_abs15','KDs absorbed / 15',2,1),('td15','Takedowns / 15',2,0),('td_acc','Takedown accuracy',3,0),
('td_def','Takedown defense',3,0),('ctrl_pct','Control time share',3,0),('ctrl_against_pct','Controlled by opp.',3,1),('sub15','Sub attempts / 15',2,0),('gnd_land15','Ground strikes / 15',2,0),
('leg_pm','Leg kicks / min',2,0),('clinch_share','Clinch strike share',3,0),('late_output_ratio','Late-round output',2,0),('ko_win_share','KO share of wins',3,0),('sub_win_share','Sub share of wins',3,0),('dec_win_share','Decision share of wins',3,0),('ko_loss_rate','KO-loss rate',3,1)]
v=lambda x,r=2: None if pd.isna(x) else round(float(x),r)
out=[]
for _,r in d.iterrows():
    out.append({"id":r.fighter_id,"n":r.NAME,"k":"" if pd.isna(r.NICKNAME) else r.NICKNAME,"d":r.DIV,
      "rec":f"{int(r.W)}-{int(r.L)}"+(f"-{int(r.D)}" if r.D else "")+(f" ({int(r.NC)} NC)" if r.NC else ""),
      "age":v(r.AGE,0),"h":v(r.HEIGHT_IN,0),"re":v(r.REACH_IN,0),"st":"" if pd.isna(r.STANCE) else r.STANCE,
      "l5":r.LAST5,"last":str(r.LAST_UFC)[:10],"uf":int(r.UFC_FIGHTS),"min":v(r.RAW_MIN,0),
      "ko":int(r.KO_W),"sb":int(r.SUB_W),"dc":int(r.DEC_W),"kol":int(r.KO_L),"sbl":int(r.SUB_L),
      "p":r.primary,"s":r.secondary,"c":r.confidence,"aio":bool(r.all_in_one_flag),"src":r.ROSTER_SOURCE,
      "sw":"" if pd.isna(r.strengths) else r.strengths,"wk":"" if pd.isna(r.weaknesses) else r.weaknesses,
      "f":[v(r['fit_'+a],0) for a in A],"x":[[v(r[k],dp),v(r['pct_'+k],2)] for k,_,dp,_ in stats]})
meta={"archetypes":A,"stats":[[k,l,lo] for k,l,_,lo in stats],"asof":"2026-10-25","n":len(out)}
data=json.dumps({"meta":meta,"f":out},separators=(',',':')).replace('</','<\\/')
t=open('explorer_template.html',encoding='utf-8').read()
open('ufc-style-index.html','w',encoding='utf-8').write(t.replace('__DATA__',data))
print(len(out))
