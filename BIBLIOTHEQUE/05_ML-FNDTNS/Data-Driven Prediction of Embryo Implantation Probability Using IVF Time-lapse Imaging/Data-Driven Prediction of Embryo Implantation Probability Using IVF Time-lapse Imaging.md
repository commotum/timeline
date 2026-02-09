# Data-Driven Prediction of Embryo Implantation Probability Using IVF Time-lapse Imaging

David H. Silver\*1
Martin Feder<sup>1</sup>
Yael Gold-Zamir<sup>1</sup>
Avital L. Polsky<sup>1</sup>
Shahar Rosentraub<sup>1</sup>
Efrat Shachor<sup>1</sup>
Adi Weinberger<sup>1</sup>
Pavlo Mazur<sup>2</sup>
Valery D. Zukin<sup>2</sup>
Alex M. Bronstein<sup>3,1</sup>

DAVID@EMBRYONICS.ME
MARTIN@EMBRYONICS.ME
YAEL@EMBRYONICS.ME
AVITAL@EMBRYONICS.ME
SHAHAR@EMBRYONICS.ME
EFRAT@EMBRYONICS.ME
ADI@EMBRYONICS.ME
P.MAZUR@IVF.COM.UA
V.ZUKIN@IVF.COM.UA
BRON@CS.TECHNION.AC.IL

#### Abstract

The process of fertilizing a human egg outside the body in order to help those suffering from infertility to conceive is known as in vitro fertilization (IVF). Despite being the most effective method of assisted reproductive technology (ART), the average success rate of IVF is a mere 20-40%. One step that is critical to the success of the procedure is selecting which embryo to transfer to the patient, a process typically conducted manually and without any universally accepted and standardized criteria. In this paper we describe a novel data-driven system trained to directly predict embryo implantation probability from embryogenesis time-lapse imaging videos. Using retrospectively collected videos from 272 embryos, we demonstrate that, when compared to an external panel of embryologists, our algorithm results in a 12% increase of positive predictive value and a 29% increase of negative predictive value.

Keywords: Deep Learning, In Vitro Fertilization, Embryo Selection, Video Classification.

# 1. Introduction

In vitro fertilization (IVF) is a procedure in which ova (egg cells) harvested from an adult female are fertilized by live sperm in vitro. After successful fertilization, the resulting embryos are incubated for several days while a trained embryologist manually tracks their development, using morphological and/or morphokinetic characteristics to generate a grade for each embryo indicative of its viability and likelihood of successful uterine implantation and, hopefully, live birth.

Although manual morphological annotation and quality assessment of embryos fertilized *in vitro* remains the gold standard for predicting IVF success, efforts to standardize

<sup>&</sup>lt;sup>1</sup> Embryonics, Tel Aviv, Israel

<sup>&</sup>lt;sup>2</sup> Clinic of Reproductive Medicine 'Nadiya', Kyiv, Ukraine

<sup>&</sup>lt;sup>3</sup> Department of Computer Science, Technion – Israel Institute of Technology, Haifa, Israel

<sup>\*</sup>Corresponding author

and improve prediction accuracy have become increasingly computational (several reviews have been published discussing such approaches from various points of view (Simopoulou et al., 2018b,a; Del Gallego et al., 2019; Liu et al., 2019; Basile et al., 2015)). Most algorithms developed for embryo outcome prediction require user-defined input parameters (such as specific morphological characteristics), execute a series of user-defined tasks, and then produce an estimated probability of achieving a user-defined outcome. Essentially, this approach can be seen as an attempt to mimic the human embryologist. While algorithms of this nature may help embryologists to more efficiently assess embryo quality, they are limited in their ability to improve outcomes as they are often dependent on the same scoring parameters as manual assessment, which is highly variable between observers (Khosravi et al., 2019; Adolfsson et al., 2018; Adolfsson and Andershed, 2018; Uyar et al., 2015; Paternot et al., 2011; Martínez-Granados et al., 2018). Lack of standardization and agreement on criteria likely contribute to the low success rate of IVF. Researchers in the assisted reproductive technology (ART) community have, therefore, increasingly turned to machine learning techniques in recent years (Simopoulou et al., 2018b; Liu et al., 2019; Curchoe and Bormann, 2019; Wang et al., 2019; Zaninovic et al., 2019).

We introduce a novel machine learning algorithm, referred to as *Ubar*, that takes timelapse images as the input and predicts embryo implantation probability. We compared the implantation probability predictions of the algorithm to embryo grades provided by an external panel of embryologists and to the known ground truth implantation results.

## 2. Data

Our dataset consisted of 8,789 retrospectively collected time-lapse videos of developing embryos, 4,087 of which were graded by an external panel of embryologists. Of the transferred embryos with known implantation data (KID), 216 were assigned the label of successful implantation (transfers that resulted in the detection of a gestational sac and fetal heartbeat at 7 and 12 weeks gestation). 56 embryos were assigned the label of failed implantation (no detection of gestational sac).

# 3. Methods

A CNN autoencoder was trained with the  $L_2$  loss on the individual frames from the unlabeled videos. The encoder comprising 10 layers was used to produce a 968-dimensional embedding per frame. An LSTM network was trained on the 4,087 graded videos receiving the embeddings of the sequence of frames and predicting the embryologist grade distribution.

The same network was used with a different binary head to predict the implantation probability on the 272 videos with known implantation data. Embryologist-graded and KID data were structured as 10 cross-validation folds, assuring no inclusion of the same patient data into training or validation sets. In order to compare UBar performance to current embryo selection standards, an external panel of five embryologists from various countries (India, Latvia, Ukraine, and the United States) assigned each embryo video a grade between 1 and 5, with 1-2 corresponding to the recommendation not to transfer due to poor embryo

quality, while 4-5 being a recommendation to transfer due to the perceived high likelihood of successful implantation.

# 4. Results

Receiver operating characteristic (ROC) curves were calculated for both UBar predictions and panel scores, with thresholds between 0 and 1 (UBar) or 1 and 5 (panel) and are depicted in Figure 1A. The area under the curve (AUC) of UBar was  $0.82 \pm 0.07$ , outperforming the expert panel (AUC =  $0.58 \pm 0.04$ ). Means and standard deviation for UBar were computed using bootstrapping over 1000 repetitions. In order to achieve a more clinically-relevant assessment of UBar's performance, the positive (PPV) and negative (NPV) predictive values were calculated for UBar predictions and compared to those of the expert panel grades (Figure 1B). PPV corresponds to the number of embryos correctly predicted as successful implantation divided by the total number of embryos predicted as failed implantation divided by the total number of embryos correctly predicted as failed implantation divided by the total number of embryos predicted to fail. Both the PPV (93%) and NPV (58%) of UBar significantly exceeded the corresponding values of the expert panel (81  $\pm$  1% and 23  $\pm$  8%, respectively), implying that application of UBar in a clinical setting could potentially improve embryo transfer outcomes.

![](_page_2_Figure_4.jpeg)

Figure 1: A. Performance of UBar compared to an expert panel of embryologists. B. Predictive values of UBar, expert panel, and a random model of which the values correlate to the prevalence of each class in the dataset: 79% successfully implanted and 21% failed implantation.

## 5. Discussion

A previously published study by (Tran et al., 2019) showed that time-lapse imaging files could be used for implantation probability prediction. However, the negatively labeled samples in Tran et al.'s study included embryos that were intentionally deselected from embryo transfer, effectively predicting a different set of outcomes: the embryologists' decisions as

well as implantation probability. Including the embryologists' decisions in the outcome prediction is arguably an easier task, as their decisions are based on designated parameters (though such parameters differ between individuals), whereas the parameters that lead to successful and failed implantation are not well understood. Furthermore, increased sample sizes of training sets have been shown to improve AUC values (Stiglic et al., 2009; Wu et al., 2018), possibly contributing to the high AUC reported by Tran et al. (0.93), whose model was trained on videos from more than 10,000 embryos.

In this paper we show that, using a small number of labeled samples, we built an embryo outcome prediction model that outperforms a panel of expert embryologists. Future directions for this model include application to a larger amount of samples originating from multiple IVF clinics. Additionally, in an effort to further improve results, we are exploring variants of the neural network, such as: inclusion of additional clinical data or training the network as a whole (multi-task network training of both the auto-encoder and the classifier).

# Acknowledgments

This work was partially supported by the Israel Innovation Authority, grant #65201. We thank D. E. Fordham for critical reading of the manuscript, and A. Gershenfeld for assistance with organizing the data.

## References

Emma Adolfsson and Anna Nowosad Andershed. Morphology vs morphokinetics: a retrospective comparison of interobserver and intra-observer agreement between embryologists on blastocysts with known implantation outcome. *JBRA Assisted Reproduction*, 2018. ISSN 1518-0557. doi: 10.5935/1518-0557.20180042. URL http://www.gnresearch.org/doi/10.5935/1518-0557.20180042. [Online; accessed 2019-08-04].

Emma Adolfsson, Sandra Porath, and Anna Nowosad Andershed. External validation of a time-lapse model; a retrospective study comparing embryo evaluation using a morphokinetic model to standard morphology with live birth as endpoint. *JBRA Assisted Reproduction*, 2018. ISSN 1518-0557. doi: 10.5935/1518-0557.20180041. URL http://www.gnresearch.org/doi/10.5935/1518-0557.20180041. [Online; accessed 2019-05-21].

N. Basile, P. Vime, M. Florensa, B. Aparicio Ruiz, J. A. García Velasco, J. Remohí, and M. Meseguer. The use of morphokinetics as a predictor of implantation: a multicentric study to define and validate an algorithm for embryo selection. *Human Reproduction* (Oxford, England), 30(2):276–283, 2 2015. ISSN 1460-2350. doi: 10.1093/humrep/deu331. PMID: 25527613.

Carol Lynn Curchoe and Charles L. Bormann. Artificial intelligence and machine learning for human reproduction and embryology presented at asrm and eshre 2018. Journal of Assisted Reproduction and Genetics, 1 2019. ISSN 1058-0468, 1573-7330. doi: 10.1007/s10815-019-01408-x. URL http://link.springer.com/10.1007/s10815-019-01408-x. [Online; accessed 2019-04-29].

- Raquel Del Gallego, José Remohí, and Marcos Meseguer. Time-lapse imaging: The state of the art. *Biology of Reproduction*, 2019. ISSN 1529-7268. doi: 10.1093/biolre/ioz035. PMID: 30810735.
- Pegah Khosravi, Ehsan Kazemi, Qiansheng Zhan, Jonas E. Malmsten, Marco Toschi, Pantelis Zisimopoulos, Alexandros Sigaras, Stuart Lavery, Lee A. D. Cooper, Cristina Hickman, Marcos Meseguer, Zev Rosenwaks, Olivier Elemento, Nikica Zaninovic, and Iman Hajirasouliha. Deep learning enables robust assessment and selection of human blastocysts after in vitro fertilization. *npj Digital Medicine*, 2(1):21, 12 2019. ISSN 2398-6352. doi: 10.1038/s41746-019-0096-y.
- Yanhe Liu, Denny Sakkas, Masoud Afnan, and Phillip Matson. Time-lapse videography for embryo selection/de-selection: a bright future or fading star? *Human Fertility*, pages 1–7, 4 2019. ISSN 1464-7273, 1742-8149. doi: 10.1080/14647273.2019.1598586.
- Luis Martínez-Granados, María Serrano, Antonio González-Utor, Nereyda Ortiz, Vicente Badajoz, María Luisa López-Regalado, Montserrat Boada, Jose A. Castilla, and Special Interest Group in Quality of ASEBIR (Society for the Study of Reproductive Biology). Reliability and agreement on embryo assessment: 5 years of an external quality control programme. Reproductive Biomedicine Online, 36(3):259–268, 3 2018. ISSN 1472-6491. doi: 10.1016/j.rbmo.2017.12.008. PMID: 29339017.
- Goedele Paternot, Alex M. Wetzels, Fabienne Thonon, Anne Vansteenbrugge, Dorien Willemen, Johanna Devroe, Sophie Debrock, Thomas M. D'Hooghe, and Carl Spiessens. Intraand interobserver analysis in the morphological assessment of early stage embryos during an ivf procedure: a multicentre study. *Reproductive biology and endocrinology*, 9: 127, 9 2011. ISSN 1477-7827. doi: 10.1186/1477-7827-9-127. PMID: 21920032 PMCID: PMC3181205.
- Mara Simopoulou, Konstantinos Sfakianoudis, Nikolaos Antoniou, Evangelos Maziotis, Anna Rapani, Panagiotis Bakas, George Anifandis, Theodoros Kalampokas, Stamatis Bolaris, Agni Pantou, Konstantinos Pantos, and Michael Koutsilieris. Making ivf more effective through the evolution of prediction models: is prognosis the missing piece of the puzzle? Systems Biology in Reproductive Medicine, 64(5):305–323, 10 2018a. ISSN 1939-6376. doi: 10.1080/19396368.2018.1504347. PMID: 30088950.
- Mara Simopoulou, Konstantinos Sfakianoudis, Evangelos Maziotis, Nikolaos Antoniou, Anna Rapani, George Anifandis, Panagiotis Bakas, Stamatis Bolaris, Agni Pantou, Konstantinos Pantos, and Michael Koutsilieris. Are computational applications the "crystal ball" in the ivf laboratory? the evolution from mathematics to artificial intelligence. *Journal of Assisted Reproduction and Genetics*, 35(9):1545–1557, 9 2018b. ISSN 1058-0468, 1573-7330. doi: 10.1007/s10815-018-1266-6.
- Gregor Stiglic, Simon Kocbek, and Peter Kokol. Comprehensibility of classifiers for future microarray analysis datasets, 2009.
- D Tran, S Cooke, PJ Illingworth, and DK Gardner. Deep learning as a predictive tool for fetal heart pregnancy following time-lapse incubation and blastocyst transfer. *Human Reproduction*, 34(6):1011–1018, 2019.

- Asli Uyar, Ayse Bener, and H. Nadir Ciray. Predictive modeling of implantation outcome in an in vitro fertilization setting: An application of machine learning methods. *Medical Decision Making: An International Journal of the Society for Medical Decision Making*, 35 (6):714–725, 2015. ISSN 1552-681X. doi: 10.1177/0272989X14535984. PMID: 24842951.
- Renjie Wang, Wei Pan, Lei Jin, Yuehan Li, Yudi Geng, Chun Gao, Gang Chen, Hui Wang, Ding Ma, and Shujie Liao. Artificial intelligence in reproductive medicine. *Reproduction*, 4 2019. ISSN 1470-1626, 1741-7899. doi: 10.1530/REP-18-0523. URL https://rep.bioscientifica.com/view/journals/rep/aop/rep-18-0523.xml. [Online; accessed 2019-07-21].
- Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S Pappu, Karl Leswing, and Vijay Pande. Moleculenet: a benchmark for molecular machine learning. *Chemical science*, 9(2):513–530, 2018.
- Nikica Zaninovic, Olivier Elemento, and Zev Rosenwaks. Artificial intelligence: its applications in reproductive medicine and the assisted reproductive technologies. *Fertility and Sterility*, 112(1):28–30, 7 2019. ISSN 00150282. doi: 10.1016/j.fertnstert.2019.05.019.