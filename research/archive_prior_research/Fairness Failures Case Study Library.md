# **Fairness Failures Case Study Library Executive Summary**

This comprehensive case study library documents eight major real-world fairness failures across healthcare, financial services, hiring systems, and government services from the past decade. These cases collectively affected millions of people, resulted in billions in settlements and remediation, and demonstrate both algorithmic and human-driven bias patterns. Each case provides detailed technical analysis, impact assessment, and mitigation strategies to serve as learning resources for preventing future discrimination.

## **Healthcare Triage Bias Case Studies**

### **Case 1: Optum Impact Pro Algorithm Racial Bias (2019)**

**Organization and Timeline:** UnitedHealth Group's Optum subsidiary deployed the Impact Pro algorithm from approximately 2010-2019, affecting 200 million Americans annually across 70+ million patients in multiple health systems before bias was discovered and published by UC Berkeley researchers in October 2019\.

**Technical Root Cause:** The algorithm used healthcare expenditure as a proxy for health need, embedding historical spending disparities where Black patients spent $1,800 less annually than equally-sick white patients due to systemic access barriers. This flawed proxy meant the algorithm systematically assigned lower risk scores to Black patients despite having more chronic conditions—at equal risk scores, Black patients had 48,772 additional chronic conditions compared to white patients in the national dataset.

**Documented Impact:** Black patients received **50% less care management services** than warranted by actual health status. Only 17.7% of Black patients received additional care versus the 46.5% who should have based on actual health needs. Black patients with equal risk scores had significantly worse clinical indicators including higher blood pressure, more severe diabetes, worse kidney function, and higher cancer rates.

**Organizational and Systemic Factors:** The failure stemmed from lack of bias testing during development, insufficient diversity in development teams, over-reliance on cost-based metrics without considering health equity implications, and limited oversight of algorithmic decision-making. Systemically, this reflected industry-wide practices affecting approximately 200 million Americans who were evaluated by similar cost-based algorithms.

**Mitigation Implemented:** Optum collaborated with researchers to develop less biased algorithms achieving 84% reduction in racial bias by incorporating both cost and health status indicators. New York State launched regulatory investigations demanding proof of anti-discrimination compliance. The case prompted introduction of the Algorithmic Accountability Act requiring companies to audit AI systems for bias.

**Key Lessons:** Healthcare costs are not race-neutral proxies for health need. Algorithms trained on biased historical data amplify existing inequities. Industry-wide problems require systemic solutions beyond individual company fixes. Continuous monitoring across racial groups is essential for detecting hidden algorithmic bias.

### **Case 2: Race-Based Kidney Function Testing (eGFR) Bias (1999-2021)**

**System-Wide Implementation:** This was a systemic issue across the entire US healthcare system using MDRD and CKD-EPI equations for kidney function estimation. The algorithms included a race coefficient that automatically increased calculated kidney function by 16-18% for Black patients based on flawed 1990s research assuming universal differences in muscle mass.

**Technical Failure Analysis:** The algorithm treated race as a biological rather than social construct, using binary racial classification that ignored genetic diversity. The race coefficient lacked scientific validation and was based on limited sample sizes from outdated research. This meant Black patients' kidney function was artificially inflated, delaying chronic kidney disease diagnosis, postponing specialist referrals, and creating barriers to transplant eligibility.

**Population-Level Impact:** Millions of Black patients over two decades experienced delayed diagnosis and treatment. Black patients have 2.9 times higher age-adjusted rate of end-stage renal disease yet faced reduced transplant eligibility due to artificially inflated kidney function scores. The algorithm systematically disadvantaged the population most affected by kidney disease.

**Reform Process:** The National Kidney Foundation and American Society of Nephrology formed a joint Task Force in July 2020, conducting a 10-month comprehensive review. The CKD-EPI 2021 equation was adopted without race coefficient, with major medical centers transitioning by 2022\. Laboratory reporting systems were updated nationwide, and the Organ Procurement and Transplant Network updated policies accordingly.

**Critical Insights:** Race is not a valid biological variable for clinical algorithms. Medical practices based on outdated research can perpetuate harm for decades. Regular reassessment of clinical tools for bias is essential. Patient advocacy and community pressure are crucial for driving change in medical practices.

## **Financial Services Discrimination Case Studies**

### **Case 3: Wells Fargo Discriminatory Lending Practices (2004-2022)**

**Scale and Timeline:** Wells Fargo, the largest residential mortgage originator in the US at the time, engaged in multiple discriminatory practices from 2004-2022, resulting in settlements of $184.3 million in 2012 and $3.7 billion in 2022 (the largest CFPB penalty in history).

**Discriminatory Mechanisms:** The bank systematically steered 34,000+ qualified African-American and Hispanic borrowers into subprime mortgages when similarly qualified white borrowers received prime loans. They charged 30,000+ minority borrowers higher fees and rates than white borrowers with similar credit profiles. Mortgage modification evaluation algorithms contained coding errors that overstated attorney fees, causing 3,200 wrongful foreclosures from 2011-2018.

**Quantified Financial Harm:** Black borrowers in Chicago paid an average of $2,937 more than white applicants. In Miami, a Black borrower seeking a $300,000 loan paid an extra $3,657 "racial surtax." The 2022 settlement addressed harm to over 16 million consumer accounts, including $246 million in remediation for nearly 850,000 wrongfully repossessed auto loan accounts.

**Root Cause Analysis:** Technical failures included algorithmic coding errors and flawed system testing. Organizationally, the bank allowed subjective pricing discretion without adequate monitoring and had weak compliance protocols. Systemically, historical redlining patterns were embedded in market practices, and wealth gaps made minority borrowers more vulnerable to predatory targeting.

**Regulatory Response and Reforms:** Multiple consent orders required comprehensive remediation programs. The Federal Reserve imposed an asset growth cap (still in effect). Wells Fargo eliminated discretionary pricing elements, implemented enhanced fair lending monitoring, and required board-level oversight of compliance obligations.

### **Case 4: Ally Financial Auto Lending Discrimination (2011-2013)**

**Scope of Discrimination:** Ally Financial, one of the largest indirect auto lenders, systematically charged higher interest rate markups to 235,000+ African-American, Hispanic, and Asian/Pacific Islander borrowers through its network of 12,000+ dealers nationwide from April 2011 through December 2013\.

**Mechanism of Bias:** The risk-based pricing system allowed unlimited dealer discretion in interest rate markups with no algorithmic controls to prevent discriminatory patterns. The compensation structure directly rewarded dealers for higher markups regardless of borrower characteristics. Minority borrowers paid $200-300 extra per loan on average, based solely on race/national origin rather than creditworthiness.

**Systemic Factors:** The auto lending market structure embedded discretionary pricing practices. Historical wealth gaps limited minority borrowers' negotiating power. Lack of transparency in auto financing made discrimination detection difficult for consumers.

**Settlement and Industry Impact:** The $98 million settlement ($80 million compensation \+ $18 million penalty) was the largest auto lending discrimination settlement at the time. It established a template for CFPB auto lending enforcement and prompted industry-wide reevaluation of dealer markup practices. Ally was required to implement statistical analysis of markup patterns, dealer training programs, and consider eliminating dealer markups entirely.

## **Hiring System Bias Case Studies**

### **Case 5: Amazon's AI Recruiting Tool Gender Bias (2014-2017)**

**Development and Discovery:** Amazon's Edinburgh office developed an AI recruiting tool with approximately 12 engineers from 2014-2017. The system created 500 computer models trained to recognize over 50,000 parameters from successful candidates' resumes. Bias was discovered by 2015, and the project was disbanded by early 2017\.

**Manifestation of Gender Discrimination:** The system penalized resumes containing "women's" (e.g., "women's rugby team captain"), downgraded graduates from all-women's colleges, and favored masculine-coded verbs like "executed" and "captured." It consistently gave lower scores to female candidates on its 1-5 star rating scale.

**Training Data Problem:** The system was trained on 10 years of resumes (2004-2014) from Amazon's male-dominated workforce (63% male). The algorithm learned patterns reflecting existing gender imbalances in tech rather than actual job performance predictors, effectively creating 100% bias against women in technical positions.

**Lessons and Industry Impact:** The case demonstrated that AI systems trained on biased historical data replicate and amplify discrimination. Even after attempting to neutralize terms like "women's," engineers couldn't guarantee the system wouldn't find other discriminatory patterns. The failure became a cautionary tale widely cited in AI ethics discussions and influenced development of bias auditing practices.

### **Case 6: HireVue AI Video Interview Discrimination (2019-2024)**

**Platform Scale:** HireVue's video interview platform was used by 700+ companies including Unilever, GE, Delta, and Oracle, conducting over 19 million video interviews with over 1 million applicants assessed by 2019\.

**Discriminatory Technology:** The system used facial expression analysis (discontinued 2021), voice pattern recognition, and behavioral assessment algorithms to generate "employability scores." Facial analysis comprised up to 30% of scores before discontinuation. The technology showed bias against people with disabilities, non-white applicants, those with autism or craniofacial conditions, and non-native English speakers.

**Specific Case Example:** D.K., an Indigenous and Deaf woman at Intuit, was denied promotion despite positive performance reviews. HireVue's automated speech recognition generated artificially low scores, and she received feedback to work on "effective communication" despite previous positive customer feedback.

**Legal and Regulatory Actions:** EPIC filed an FTC complaint in November 2019\. The ACLU filed EEOC and Colorado Civil Rights Division complaints in March 2024\. Multiple lawsuits were filed under biometric privacy laws. HireVue discontinued facial analysis in January 2021 but continues using speech analysis despite documented bias concerns.

**Critical Assessment:** Princeton Professor Arvind Narayanan called it "AI whose only conceivable purpose is to perpetuate societal biases." The case highlighted that biometric analysis of job fitness lacks scientific validity and that third-party vendors can expose client companies to discrimination liability.

## **Government Services Bias Case Studies**

### **Case 7: Australia's Robodebt Scheme (2016-2020)**

**Unprecedented Scale of Harm:** The automated debt assessment system issued **470,000 unlawful debt notices** to welfare recipients, wrongfully recovering A$746 million from 381,000 individuals. A$1.75 billion in debts were ultimately written off following class action.

**Fatal Algorithm Flaw:** The system used illegal "income averaging" methodology, dividing annual employer-reported income evenly across fortnights while ignoring employment gaps and variable work patterns. This violated Social Security Act requirements for actual fortnightly income assessment. The system reversed the burden of proof, automatically generating debts that recipients had to disprove.

**Human Cost:** The Royal Commission documented suicides linked to debt notices, with 663 vulnerable people (with mental illness, abuse histories) dying shortly after receiving notices. Average wrongful debt was A$1,958 per person. Many victims borrowed money or sold possessions to pay unlawful debts.

**Governance Failures:** Senior public servants misled Cabinet about the scheme's legal basis and omitted legal advice warning of unlawfulness. The scheme eliminated human adjudication that previously caught errors. Political pressure for A$1.7 billion in budget savings overrode legal and ethical considerations.

**Comprehensive Reform:** The scheme was terminated in May 2020 with full debt forgiveness. A A$1.8 billion class action settlement compensated affected individuals. The Royal Commission made 57 recommendations including human-centered design requirements, mandatory transparency for algorithm logic, and independent oversight of government AI systems.

### **Case 8: Michigan's MiDAS Unemployment Fraud Detection (2013-2015)**

**Catastrophic Error Rate:** The $47 million Michigan Integrated Data Automated System falsely accused 42,000+ individuals of unemployment fraud with a **93% error rate** in fraud determinations according to court findings. 34,000+ confirmed wrongful accusations required remediation.

**Automated Injustice:** The "robo-adjudication" process made fully automated fraud determinations with no human review. The system deferred to employer claims that workers "quit" when they were actually laid off. Missing or corrupt data was used as basis for fraud findings. UIA eliminated over 400 fraud investigators simultaneously with MiDAS deployment.

**Financial Devastation:** Individual fines commonly ranged from $10,000-$50,000 per wrongfully accused person. **11,000 families filed bankruptcy** directly attributable to false accusations. The state seized wages and tax refunds without due process. UIA coffers increased from $3.1 million (2011) to $155 million (2016) through false fraud penalties.

**Legal Remediation:** Federal courts intervened through class action lawsuits. The Michigan legislature passed laws in 2017 requiring manual fraud determinations and eliminating "robo-adjudication." A $20 million settlement was approved in 2024 for wrongfully accused claimants. The Michigan Supreme Court allowed constitutional claims to proceed against the state.

**Systemic Lessons:** The case demonstrated that fully automated adjudication of complex benefits decisions is inherently flawed. Revenue generation incentives led to systemic bias toward false accusations. Neither state officials nor vendors took adequate responsibility for algorithmic failures.

## **Cross-Domain Analysis and Universal Lessons**

### **Common failure patterns across all domains**

These case studies reveal consistent patterns of algorithmic and systemic bias: automation without adequate human oversight in high-stakes decisions, reverse burden of proof requiring citizens to disprove algorithmic determinations, disproportionate impact on vulnerable populations with limited capacity to contest decisions, and revenue or cost-saving incentives overriding accuracy and fairness considerations.

### **Technical commonalities**

Across healthcare, finance, employment, and government services, we see algorithms trained on biased historical data perpetuating discrimination, use of flawed proxy variables that embed systemic inequities, lack of diverse representation in training data and development teams, and inadequate testing and validation before deployment at scale.

### **Organizational failures**

Every case demonstrated insufficient bias testing and monitoring protocols, weak governance and accountability structures, prioritization of efficiency over equity considerations, and limited transparency preventing meaningful oversight or challenge.

### **Mitigation strategies that work**

Successful remediation has required mandatory human review of algorithmic decisions, enhanced transparency and explainability requirements, regular bias auditing and monitoring across protected classes, strong regulatory enforcement with significant penalties, and involvement of affected communities in system design and oversight.

### **Future prevention requirements**

These cases collectively demonstrate the need for comprehensive algorithmic accountability legislation, independent oversight bodies for government and high-stakes AI systems, mandatory bias impact assessments before deployment, continuous monitoring and adjustment requirements, strong whistleblower protections for those identifying bias, and meaningful remedies for individuals harmed by algorithmic discrimination.

## **Conclusion**

This case study library documents how algorithmic and systemic bias has caused massive harm across critical domains of modern life. The patterns are clear: when organizations deploy automated systems without adequate safeguards, testing, and oversight, they perpetuate and amplify existing societal inequities. The human cost—measured in wrongful denials of healthcare, discriminatory lending practices, biased hiring decisions, and unjust government actions—demonstrates that fairness failures are not merely technical problems but fundamental challenges to justice and human dignity. These cases provide essential lessons for developing more equitable systems that serve all members of society fairly.

