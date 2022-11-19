library(ARTool)
setwd("/Users/yiqinpan/OneDrive - University of Florida/research/CAT Detection/detect_preknowledge_in_cat_codes/detection")
types = c('false_neg_item', 'false_neg_ppl', 'false_posi_item', 'false_posi_ppl', 'precision_item', 'precision_ppl')
for (ele in types){
  data = read.csv(file = paste('result_mid/',ele,".csv",sep = ""),header = FALSE)[,c(5,6,7,8,9)]
  colnames(data)=c("ewp_rate", "ci_rate", "iterat_times", "ab_cri", "val")
  data = data.frame(data)
  data$ewp_rate = factor(data$ewp_rate) 
  data$ci_rate = factor(data$ci_rate) 
  data$iterat_times = factor(data$iterat_times) 
  data$ab_cri = factor(data$ab_cri) 
  
  data=data[which(data$ewp_rate!=0 & data$iterat_times!=20& data$iterat_times!=40),]
  model = art(val ~ ewp_rate*ci_rate*ab_cri*iterat_times, data = data)
  #print(model)
  
  Result = anova(model)
  Result$part.eta.sq = with(Result, `Sum Sq`/(`Sum Sq` + `Sum Sq.res`))
  print(ele)
  print(Result)
  #write.csv(Result, file =  paste('anova/',ele,".csv",sep = ""))


}