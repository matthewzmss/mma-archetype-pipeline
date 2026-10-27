import pandas as pd, json, sys
from pathlib import Path
d=pd.read_csv('out/fighter_archetypes.csv')
A=["Brawler","Pressure Fighter","Point Fighter","Sniper","Defensive Technician","Submission Threat","Grappler / Wrestler","Point-Fighting Wrestler","Ground-and-Pound Wrestler","All-in-One"]
stats=[('slpm','Sig. landed / min',2,0),('sapm','Sig. absorbed / min',2,1),('sig_acc','Strike accuracy',3,0),('sig_def','Strike defense',3,0),('str_diff','Strike differential',2,0),
('kd15','Knockdowns / 15',2,0),('kd_per_100_landed','KDs per 100 landed',2,0),('kd_abs15','KDs absorbed / 15',2,1),('td15','Takedowns / 15',2,0),('td_acc','Takedown accuracy',3,0),
('td_def','Takedown defense',3,0),('ctrl_pct','Control time share',3,0),('ctrl_against_pct','Controlled by opp.',3,1),('sub15','Sub attempts / 15',2,0),('gnd_land15','Ground strikes / 15',2,0),
('leg_pm','Leg kicks / min',2,0),('clinch_share','Clinch strike share',3,0),('late_output_ratio','Late-round output',2,0),('ko_win_share','KO share of wins',3,0),('sub_win_share','Sub share of wins',3,0),('dec_win_share','Decision share of wins',3,0),('ko_loss_rate','KO-loss rate',3,1)]
# every model metric: (key, label, group, direction 1=higher better/-1=lower better/0=style, format p=percent n=number i=count m=minutes x=ratio)
mets=[('slpm','Sig. strikes landed / min','Striking',1,'n'),('sapm','Sig. strikes absorbed / min','Striking',-1,'n'),('sig_att_pm','Sig. strikes attempted / min','Striking',1,'n'),
('opp_sig_att_pm','Opp. sig. attempts / min','Striking',0,'n'),('total_str_pm','Total strikes attempted / min','Striking',1,'n'),('sig_acc','Sig. strike accuracy','Striking',1,'p'),
('sig_def','Sig. strike defense','Striking',1,'p'),('str_diff','Strike differential / min','Striking',1,'n'),('kd15','Knockdowns / 15 min','Striking',1,'n'),
('kd_abs15','Knockdowns absorbed / 15 min','Striking',-1,'n'),('kd_per_100_landed','Knockdowns per 100 landed','Striking',1,'n'),
('clean_ko_pts','Clean KO points / fight','Knockout power',1,'n'),('early_ko_rate','Early KO/TKO credit / fight','Knockout power',1,'n'),
('ko_per_100_landed','KO wins per 100 landed','Knockout power',1,'n'),
('dist_att_open_pm','Distance attempts / open min','Distance striking',1,'n'),('dist_land_open_pm','Distance landed / open min','Distance striking',1,'n'),
('dist_abs_open_pm','Distance absorbed / open min','Distance striking',-1,'n'),('dist_acc','Distance accuracy','Distance striking',1,'p'),('dist_def','Distance defense','Distance striking',1,'p'),
('head_share','Head share of sig. landed','Target & position',0,'p'),('body_share','Body share of sig. landed','Target & position',0,'p'),('leg_share','Leg share of sig. landed','Target & position',0,'p'),
('leg_pm','Leg strikes landed / min','Target & position',1,'n'),('clinch_share','Clinch share of sig. landed','Target & position',0,'p'),('ground_share','Ground share of sig. landed','Target & position',0,'p'),
('open_time_pct','Time at open distance','Target & position',0,'p'),
('td15','Takedowns / 15 min','Grappling',1,'n'),('td_att15','Takedown attempts / 15 min','Grappling',1,'n'),('td_acc','Takedown accuracy','Grappling',1,'p'),('td_def','Takedown defense','Grappling',1,'p'),
('opp_td_att15','Opp. takedown attempts / 15 min','Grappling',0,'n'),('sub15','Sub attempts / 15 min','Grappling',1,'n'),('opp_sub15','Opp. sub attempts / 15 min','Grappling',-1,'n'),
('rev15','Reversals / 15 min','Grappling',1,'n'),('ctrl_pct','Control time share','Grappling',1,'p'),('ctrl_against_pct','Controlled by opponent','Grappling',-1,'p'),
('gnd_land15','Ground strikes landed / 15 min','Grappling',1,'n'),('gnd_land_per_ctrl_min','Ground strikes / control min','Grappling',1,'n'),
('late_output_ratio','Late-round output (vs R1)','Pace & durability',1,'x'),('avg_fight_min','Average fight time (min)','Pace & durability',0,'m'),('ko_loss_rate','KO-loss rate','Pace & durability',-1,'p'),
('ko_win_share','KO share of wins','Results',1,'p'),('sub_win_share','Sub share of wins','Results',1,'p'),('dec_win_share','Decision share of wins','Results',0,'p'),
('finish_win_share','Finish share of wins','Results',1,'p'),('decision_fight_rate','Fights going to decision','Results',0,'p'),('r1_finish_rate','Round-1 finish rate','Results',1,'p'),
('ko_wins','KO/TKO wins','Results',1,'i'),('sub_wins','Submission wins','Results',1,'i'),('gnp_ko_wins','Ground-and-pound KO wins','Results',1,'i'),('gnp_ko_share','GnP share of KO wins','Results',0,'p')]
# archetype breakdown pages: description + scoring recipe, mirroring score() and power_score() in archetypes.py
# items: (label, weight, +1 = higher percentile scores / -1 = lower scores, metric keys); "S" = 1 - mean(td15, ctrl_pct, sub15) percentile
WRESTLE_NOTE="Wrestler styles are scaled down unless the fighter actually wrestles: the average of their takedown-rate, takedown-attempt and control-share percentiles must pass the 45th percentile to score at all, with full credit from the 80th. A fighter can hold only one of the three wrestler styles."
recipes={
"Power":{"d":"Knockout power, rated 0–100 for every fighter on top of their style. Rewards clean one-strike knockouts, quick finishes, frequent knockdowns and needing few strikes to end a fight. 90+ is elite (top 10%).",
  "note":"A clean KO is a finish UFCStats credits to a single strike (\"Punch to Head\", \"Kick to Head\", \"Flying Knee\"). Finishes from a flurry (\"Punches to Head\") count as TKOs. Each clean KO is worth 1 point plus an early bonus. Each KO/TKO from strikes earns early credit only. Early credit starts at 1.0 at the opening bell and halves every 5 minutes of fight time. Cuts, injuries and doctor or corner stoppages don't count.","items":[
  ("Clean one-shot KOs (bonus if early)",.30,1,["clean_ko_pts"]),("Early KO/TKO finishes",.25,1,["early_ko_rate"]),("Knockdowns / 15 min",.20,1,["kd15"]),
  ("KO wins per 100 strikes landed",.15,1,["ko_per_100_landed"]),("Short fights",.10,-1,["avg_fight_min"])]},
"Brawler":{"d":"High-volume, high-risk striking. Throws a lot, takes a lot back and is happy to trade in the pocket.","items":[
  ("Sig. strikes attempted / min",.26,1,["sig_att_pm"]),("Sig. strikes absorbed / min",.18,1,["sapm"]),("Sig. strikes landed / min",.14,1,["slpm"]),("Low strike defense",.10,-1,["sig_def"]),
  ("Knockdowns / 15 min",.08,1,["kd15"]),("KO share of wins",.06,1,["ko_win_share"]),("Striking over grappling",.18,1,["S"])]},
"Pressure Fighter":{"d":"Walks opponents down and outworks them. Output holds or rises in later rounds, with clinch work and takedown attempts mixed in, and fights tend to go long.","items":[
  ("Sig. strikes landed / min",.22,1,["slpm"]),("Late-round output (vs R1)",.20,1,["late_output_ratio"]),("Sig. strikes attempted / min",.14,1,["sig_att_pm"]),("Total strikes attempted / min",.10,1,["total_str_pm"]),
  ("Strike differential / min",.10,1,["str_diff"]),("Clinch share of sig. landed",.08,1,["clinch_share"]),("Takedown attempts / 15 min",.08,1,["td_att15"]),("Long fights",.08,1,["avg_fight_min"])]},
"Point Fighter":{"d":"Wins rounds on the scorecards. Lands more than they absorb, with good accuracy and defense, and wins mostly by decision rather than knockout.","items":[
  ("Sig. strikes landed / min",.18,1,["slpm"]),("Strike differential / min",.18,1,["str_diff"]),("Sig. strike accuracy",.13,1,["sig_acc"]),("Decision share of wins",.13,1,["dec_win_share"]),
  ("Low damage absorbed",.09,-1,["sapm"]),("Sig. strike defense",.09,1,["sig_def"]),("Few KO wins",.08,-1,["ko_win_share"]),("Striking over grappling",.12,1,["S"])]},
"Sniper":{"d":"Precise, efficient striking from range. Very accurate at distance with real power per strike, and no need for high volume.","items":[
  ("Distance accuracy",.26,1,["dist_acc"]),("Strike differential / min",.18,1,["str_diff"]),("Knockdowns per 100 landed",.14,1,["kd_per_100_landed"]),("Low damage absorbed",.10,-1,["sapm"]),
  ("Low volume (picks shots)",.10,-1,["sig_att_pm"]),("KO share of wins",.08,1,["ko_win_share"]),("Striking over grappling",.14,1,["S"])]},
"Defensive Technician":{"d":"Hard to hit. Opponents throw plenty and miss, and the fighter makes them pay with a positive differential while rarely getting dropped.","items":[
  ("Distance defense",.30,1,["dist_def"]),("Low distance damage absorbed",.22,-1,["dist_abs_open_pm"]),("Strike differential / min",.14,1,["str_diff"]),("Rarely knocked down",.10,-1,["kd_abs15"]),
  ("Sig. strike accuracy",.08,1,["sig_acc"]),("Opponents throw a lot",.06,1,["opp_sig_att_pm"]),("Striking over grappling",.10,1,["S"])]},
"Submission Threat":{"d":"Always hunting the finish on the mat. High submission-attempt rate and a large share of wins by tapout.","items":[
  ("Sub attempts / 15 min",.38,1,["sub15"]),("Sub share of wins",.20,1,["sub_win_share"]),("Submission wins",.18,1,["sub_wins"]),("Takedowns / 15 min",.10,1,["td15"]),
  ("Control time share",.08,1,["ctrl_pct"]),("Reversals / 15 min",.06,1,["rev15"])]},
"Grappler / Wrestler":{"d":"Takes the fight down and keeps it there. High takedown volume and control time, hard to take down and hard to hold down.","note":WRESTLE_NOTE,"items":[
  ("Takedowns / 15 min",.28,1,["td15"]),("Control time share",.28,1,["ctrl_pct"]),("Takedown attempts / 15 min",.12,1,["td_att15"]),("Takedown accuracy",.10,1,["td_acc"]),
  ("Takedown defense",.10,1,["td_def"]),("Sub attempts / 15 min",.06,1,["sub15"]),("Rarely controlled",.06,-1,["ctrl_against_pct"])]},
"Point-Fighting Wrestler":{"d":"Wrestles to win rounds rather than finish. Plenty of takedowns and control, decision wins and relatively little damage from top.","note":WRESTLE_NOTE,"items":[
  ("Control time share",.28,1,["ctrl_pct"]),("Takedowns / 15 min",.24,1,["td15"]),("Decision share of wins",.20,1,["dec_win_share"]),("Few finishes",.10,-1,["finish_win_share"]),
  ("Few ground strikes per control min",.10,-1,["gnd_land_per_ctrl_min"]),("Takedown accuracy",.08,1,["td_acc"])]},
"Ground-and-Pound Wrestler":{"d":"Takes them down and does damage. Takedowns and control paired with heavy ground striking and knockouts from top.","note":WRESTLE_NOTE,"items":[
  ("Ground strikes landed / 15 min",.26,1,["gnd_land15"]),("Takedowns / 15 min",.22,1,["td15"]),("Control time share",.18,1,["ctrl_pct"]),("Ground strikes / control min",.12,1,["gnd_land_per_ctrl_min"]),
  ("Ground-and-pound KO wins",.12,1,["gnp_ko_wins"]),("KO share of wins",.10,1,["ko_win_share"])]},
"All-in-One":{"d":"Above the division norm everywhere: striking, wrestling, takedown defense and grappling. Reserved for fighters with no obvious hole.",
  "note":"Score = half the average of the four areas plus half the weakest one, so a single hole sinks it. Only awarded when every area ranks at or above the division's 55th percentile and their average reaches the 65th.","items":[
  ("Striking: accuracy, defense, differential",.25,1,["sig_acc","sig_def","str_diff"]),("Wrestling: takedown rate and accuracy",.25,1,["td15","td_acc"]),
  ("Takedown defense",.25,1,["td_def"]),("Grappling: sub attempts and control",.25,1,["sub15","ctrl_pct"])]},
}
v=lambda x,r=2: None if pd.isna(x) else round(float(x),r)

# per-round stats for every UFC fight of every indexed fighter, with the opponent's numbers for the same round
RK=["KD","SIG.STR._L","SIG.STR._A","TOTAL STR._L","TOTAL STR._A","TD_L","TD_A","SUB.ATT","REV.","CTRL_S","HEAD_L","HEAD_A","BODY_L","BODY_A","LEG_L","LEG_A","DISTANCE_L","DISTANCE_A","CLINCH_L","CLINCH_A","GROUND_L","GROUND_A"]
act=set(d.fighter_id)
L=pd.read_csv('out/fight_results_clean.csv')
L=L[L.FID.isin(act)].sort_values('DATE',ascending=False)
rs=pd.read_csv('out/round_stats_clean.csv.gz',usecols=['URL','RND','FID','RSEC','END_SEC','RLEN']+RK)
rs=rs[rs.URL.isin(L.URL)]
rs[RK]=rs[RK].fillna(0)
rr=rs.merge(rs[['URL','RND','FID']+RK],on=['URL','RND'],suffixes=('','_o'))
rr=rr[rr.FID!=rr.FID_o].sort_values('RND')
OK=[c+'_o' for c in RK]
rounds={}
for (fid,url),g in rr.groupby(['FID','URL']):
    rounds[(fid,url)]=[[int(s),[int(a) for a in me],[int(b) for b in op]] for s,me,op in zip(g.RSEC,g[RK].values,g[OK].values)]
fin=rs.groupby('URL').agg(es=('END_SEC','first'),rl=('RLEN','first'))
def fights(fid):
    out=[]
    for x in L[L.FID==fid].itertuples():
        e=fin.loc[x.URL] if x.URL in fin.index else None
        out.append({"dt":x.DATE,"ev":x.EVENT,"wc":x.WEIGHTCLASS,"o":x.OPP,"oid":x.OPP_FID if x.OPP_FID in act else "","r":x.RES,
          "me":x.METHOD,"de":"" if pd.isna(x.DETAILS) else x.DETAILS,"er":int(x.END_ROUND),"es":None if e is None else v(e.es,0),
          "sr":None if e is None else len(json.loads(e.rl)),"t":bool(x.TITLE),"u":x.URL.rsplit('/',1)[-1],"R":rounds.get((fid,x.URL),[])})
    return out

# UFC.com profiles (fetch_ufc_profiles.py); fighters not fetched yet simply have no photo/bio
pf=Path('ref/ufc_profiles.json')
prof=json.loads(pf.read_text()) if pf.exists() else {}
def ufc(fid):
    p=prof.get(fid)
    if not p or p.get("status")!="ok": return None
    return {"img":p.get("img"),"url":p.get("url"),"rec":(p.get("record") or "").replace(" (W-L-D)",""),"tags":p.get("tags",[]),"hero":p.get("hero",{}),"bio":p.get("bio",{})}

out=[]
for _,r in d.iterrows():
    out.append({"id":r.fighter_id,"n":r.NAME,"k":"" if pd.isna(r.NICKNAME) else r.NICKNAME,"d":r.DIV,
      "rec":f"{int(r.W)}-{int(r.L)}"+(f"-{int(r.D)}" if r.D else "")+(f" ({int(r.NC)} NC)" if r.NC else ""),
      "age":v(r.AGE,0),"h":v(r.HEIGHT_IN,0),"re":v(r.REACH_IN,0),"st":"" if pd.isna(r.STANCE) else r.STANCE,
      "l5":r.LAST5,"last":str(r.LAST_UFC)[:10],"uf":int(r.UFC_FIGHTS),"min":v(r.RAW_MIN,0),
      "ko":int(r.KO_W),"sb":int(r.SUB_W),"dc":int(r.DEC_W),"kol":int(r.KO_L),"sbl":int(r.SUB_L),"dcl":int(r.DEC_L),
      "w":int(r.W),"l":int(r.L),"r1":int(r.R1_FIN_W),"tf":int(r.TITLE_FIGHTS),"first":str(r.FIRST_UFC)[:10],
      "p":r.primary,"s":r.secondary,"c":r.confidence,"pw":v(r.power,1),"aio":bool(r.all_in_one_flag),"src":r.ROSTER_SOURCE,
      "sw":"" if pd.isna(r.strengths) else r.strengths,"wk":"" if pd.isna(r.weaknesses) else r.weaknesses,
      "f":[v(r['fit_'+a],0) for a in A],"x":[[v(r[k],dp),v(r['pct_'+k],2)] for k,_,dp,_ in stats],
      "m":[[v(r[k],3),v(r['pct_'+k],2)] for k,*_ in mets],"u":ufc(r.fighter_id),"fi":fights(r.fighter_id)})
meta={"archetypes":A,"stats":[[k,l,lo] for k,l,_,lo in stats],"mets":[list(m) for m in mets],"recipes":recipes,"rk":RK,"asof":"2026-10-25","n":len(out)}
data=json.dumps({"meta":meta,"f":out},separators=(',',':'),ensure_ascii=False).replace('</','<\\/')
t=open('explorer_template.html',encoding='utf-8').read()
open('ufc-style-index.html','w',encoding='utf-8').write(t.replace('__DATA__',data))
print(len(out),"fighters,",sum(1 for f in out if f["u"]),"with UFC.com profiles,",sum(len(f["fi"]) for f in out),"fights")
