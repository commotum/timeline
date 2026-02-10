1. **Number of distinct tasks evaluated:** 5

   Verbatim evidence:
   - "In our first evaluation, we investigate the potential of our neurosymbolic VLP framework to leverage the power of VLMs in diverse visual concept learning tasks. For this, we compare the performance of five different base VLMs with and without VLP processing on five datasets." (Section 4, RQ1)
   - "Specifically, we evaluate on the datasets Bongard-HOI [13], Bongard-OpenWorld [45] and Bongard-RWR[19], which are based on real-world images and incorporate a diverse set of visual concepts. For a real-world dataset that provides more complex logical rules, we utilize COCOLogic [37] and create 10 tasks from it, one for each class. For the synthetic dataset, we use CLEVR-Hans3 [35], where we leverage the three classes to construct three tasks with complex logical rules from it." (Section 4, Data)

2. **Number of trained model instances required to cover all tasks:** 1

   Verbatim evidence:
   - "Since VLP is training-free, most rules are assigned uniform probabilities." (Section 3.4)
   - "Notably, none of the model encoders were specifically finetuned on these datasets, demonstrating that VLP grants domain-independent flexibility." (Section 4, RQ1)
   - "Unlike task-specific DSLs, ours is VLP-specific: it defines a general symbolic interface that remains invariant across domains and tasks." (Section 3.3)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{5\ \text{tasks}}{1\ \text{model}} = 5
}
$$
