"""Create a formatted 4-page PDF technical report from project results."""
import json, csv, os
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether

ROOT=os.path.dirname(__file__); OUT=os.path.join(ROOT,'outputs')
r=json.load(open(os.path.join(OUT,'results.json')))
rows=list(csv.DictReader(open(os.path.join(OUT,'sensitivity_results.csv'))))
pdf=os.path.join(ROOT,'UAV_Strategy_Engine_Technical_Report.pdf')
doc=SimpleDocTemplate(pdf,pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=16*mm,bottomMargin=16*mm)
styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='CoverTitle',parent=styles['Title'],fontSize=21,leading=26,alignment=TA_CENTER,spaceAfter=12)); styles.add(ParagraphStyle(name='Sub',parent=styles['Normal'],alignment=TA_CENTER,textColor=colors.HexColor('#475569'),spaceAfter=8)); styles.add(ParagraphStyle(name='Small',parent=styles['BodyText'],fontSize=8.5,leading=11));
story=[]
def P(txt,style='BodyText'): story.append(Paragraph(txt,styles[style])); story.append(Spacer(1,5))
def H(txt): story.append(Paragraph(txt,styles['Heading2'])); story.append(Spacer(1,4))
def pic(name,width=165*mm):
    path=os.path.join(OUT,name)
    if os.path.exists(path):
        im=Image(path); im.drawWidth=width; im.drawHeight=width*im.imageHeight/im.imageWidth; story.append(im); story.append(Spacer(1,5))

story += [Spacer(1,20*mm), Paragraph('Genetic Algorithm Strategy Engine for UAV Disaster Survey',styles['CoverTitle']), Paragraph('Technical Report | Synthetic 50 × 50 Hazard Grid',styles['Sub']), Spacer(1,8*mm)]
P('<b>Project scope.</b> This project demonstrates a Genetic Algorithm (GA) that evolves waypoint routes, relative speed settings, and sensor power profiles for three simulated UAVs. The environment contains blocked cells, high-priority survey zones, target locations, and graded hazard pockets. The work is intended for academic simulation only; it is not a flight controller or a safety-certified path planner.')
P('<b>Team:</b> [Add three team member names] &nbsp;&nbsp; <b>Course:</b> [Add course] &nbsp;&nbsp; <b>Date:</b> [Add submission date]')
H('Abstract')
P('The mission balances conflicting objectives: increase priority-weighted coverage and target detection while reducing energy expenditure, flight time, and hazard exposure. A hybrid chromosome encodes waypoint coordinates, speed, and sensor power. Tournament selection, segment crossover, Gaussian mutation, elitism, repair, and penalty-based constraint handling evolve candidate plans. The main experiment compares the best evolved plan with random search and records a short hyperparameter sensitivity study.')
H('1. Environment and chromosome design')
P('The synthetic map is a 50 × 50 matrix. Each cell is open or blocked, may contain hazard intensity from zero to one, and may belong to a priority zone. Three UAVs begin at fixed start cells; six target positions are used for detection scoring. A chromosome has shape (3, 8, 4), with each gene represented as (x, y, v, s): grid coordinates, relative speed in [0.5, 2.0], and sensor power in [0, 1]. This hybrid real-valued representation allows smooth coordinate perturbations and meaningful waypoint-segment inheritance. It is compact, but it does not model acceleration, turning radius, communications, or full vehicle dynamics.')
H('2. Fitness function')
P('The objective is F = 0.35C + 0.25D − 0.15E_norm − 0.10T_norm − 0.15H_norm − P. C is priority-weighted coverage, D is mean target detection probability, E_norm is normalized energy, T_norm is normalized travel time, H_norm is normalized hazard exposure, and P penalizes obstacle-crossing segments and range violations. Coverage and detection receive 60% of the positive/negative weighting emphasis because the simulated mission prioritizes survey success. These weights are initial assumptions and should be varied for different mission priorities. A weighted sum returns one score and does not recover the full Pareto frontier.')
P('Repair clips coordinates to the map, moves blocked waypoints to the nearest free cell, and clamps speed and sensor genes. Routes beyond the approximate range limit are pulled toward the start. Segments are sampled for obstacle intersections and penalized. This is a practical hybrid of repair and penalty methods, but it does not guarantee every connecting segment avoids obstacles; A* repair between consecutive waypoints is recommended as future work.')
story.append(PageBreak())
H('3. Genetic operators and experimental setup')
P('The algorithm initializes a random population, selects parents by tournament, swaps a contiguous waypoint block during crossover, and occasionally inherits a whole drone route. Gaussian mutation perturbs coordinates and control settings; a small share of coordinate mutations resamples a waypoint globally. Elitism carries the top two candidates forward. The main run used population size 35, mutation rate 0.12, tournament size 3, and a maximum of 60 generations. A 25-generation stagnation criterion is implemented; the run completed the configured 60 generations. Fixed seeds make the demo reproducible in a comparable software environment.')
H('4. Main results')
data=[['Metric','Recorded result'],['Generations completed',str(r['generations_completed'])],['GA best fitness',f"{r['ga_best_fitness']:.4f}"],['Random search best fitness',f"{r['random_search_best_fitness']:.4f}"],['Random-search candidates',str(r['random_search_trials'])],['Priority-weighted coverage',f"{r['coverage']:.4f}"],['Mean target detection',f"{r['detection']:.4f}"],['Energy expenditure',f"{r['energy']:.1f}"],['Flight time',f"{r['time']:.1f}"],['Hazard exposure',f"{r['hazard']:.2f}"]]
t=Table(data,colWidths=[75*mm,70*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dbeafe')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.35,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)])); story.append(t); story.append(Spacer(1,7))
P(f"The GA's best score ({r['ga_best_fitness']:.4f}) exceeded the best random-search score ({r['random_search_best_fitness']:.4f}) under the same implemented objective. This is a single seeded comparison, so it is evidence for this run only—not proof of general superiority. Coverage was {r['coverage']:.3f}, and detection was {r['detection']:.3f}. Scores depend on the synthetic map, simplified sensor footprint, normalization, and penalty scale.")
pic('convergence.png',150*mm)
story.append(PageBreak())
H('5. Evolved plan and coverage')
pic('flight_paths.png',125*mm)
P('Figure 2. Evolved routes over the synthetic hazard and obstacle map. Colored lines distinguish the three UAVs; stars mark target locations. Route segments are evaluated by sampling and obstacle crossings receive a penalty.', 'Small')
pic('coverage_heatmap.png',125*mm)
P('Figure 3. Sensor coverage heatmap generated from the best chromosome. The displayed footprint is a simplified circular sensing model around each waypoint; it is not a calibrated sensor model.', 'Small')
story.append(PageBreak())
H('6. Hyperparameter sensitivity')
P('The sensitivity experiment ran 12 generations for each setting, with one seed per configuration. Results below are preliminary; stochastic variation and short runs mean they should not be treated as definitive parameter rankings.')
sens=[['Factor','Setting','Best fitness']]
for row in rows:
    if row['factor']=='population': setting='population='+row['population_size']
    elif row['factor']=='mutation': setting='mutation='+row['mutation_rate']
    else: setting='tournament='+row['tournament_size']
    sens.append([row['factor'],setting,f"{float(row['best_fitness']):.4f}"])
t=Table(sens,colWidths=[45*mm,65*mm,35*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dbeafe')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.35,colors.grey),('FONTSIZE',(0,0),(-1,-1),8),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)])); story.append(t); story.append(Spacer(1,6))
P('Population sizes 15, 25, and 35 scored −0.4213, −1.4793, and 0.0815 respectively. Mutation rates 0.04, 0.12, and 0.25 scored 0.1204, −0.0666, and −0.1622, so the lowest rate led this short run. Tournament sizes 2, 3, and 5 scored 0.0196, 0.1021, and 0.1018; tournament sizes 3 and 5 were nearly tied. Each setting had only one seed and 12 generations, so these findings are preliminary. Repeat each setting across at least 10 seeds before drawing robust conclusions.')
pic('random_vs_ga.png',120*mm)
H('7. Conclusion and limitations')
P('The project demonstrates a modular GA pipeline, custom chromosome, scalar multi-objective fitness, spatial operators, constraint repair, visual outputs, a random-search baseline, and sensitivity experiments. Limitations include synthetic maps, simplified sensor/energy models, penalty-based obstacle avoidance, and a single-score objective. Future work should add A* segment repair, repeated seeds, uncertainty-aware sensing, inter-UAV separation, and a Pareto method such as NSGA-II.')
H('References')
P('1. Holland, J. H. <i>Adaptation in Natural and Artificial Systems</i>. University of Michigan Press, 1975.<br/>2. Deb, K. et al. “A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.” <i>IEEE Transactions on Evolutionary Computation</i>, 6(2), 2002, pp. 182–197.<br/>3. Hart, P. E., Nilsson, N. J., and Raphael, B. “A Formal Basis for the Heuristic Determination of Minimum Cost Paths.” <i>IEEE Transactions on Systems Science and Cybernetics</i>, 4(2), 1968, pp. 100–107.', 'Small')
doc.build(story)
print(pdf)
