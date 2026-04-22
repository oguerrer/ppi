##########################################################
# Purpose: to retrieve data from statswalesr to inform ppi 
## 1a. Retrieving statswales indicator data 
## 1b. Reshaping and aggregating for modelling

#install.packages("statswalesr")
library("statswalesr")
library("tidyverse")
outputpath <- "~/wales/"

health_search <- unique(statswales_search("child")$Dataset)
health_search <- c(health_search,unique(statswales_search("health")$Dataset))
health_search <- c(health_search,unique(statswales_search("care")$Dataset))

# A loop helps us to retrieve metadata from string of datasets
health_fullmetadata <- data.frame() 
for(i in health_search$Dataset) {
  temp <- statswales_get_metadata(i)
  health_fullmetadata <- rbind(health_fullmetadata,temp)
  rm(temp)
}
# We select the Title of the indicator 
health_metadata <- unique(health_fullmetadata[health_fullmetadata$Tag_ENG=="Title", c("Dataset","Description_ENG")])

# To fetch data I create a dataframe to append rows from the loop below
# this as functions only retrieve one indicator at a time

indicators_data <- data.frame()
for(i in unique(health_metadata$Dataset)) {  
  temp <- statswales_get_dataset(i) 
  
  # retrieving relevant columns to identify unique indicators
  if(c('Dataitem_Code') %in% colnames(temp)) {  
    tempi <- unique(temp[,c('Dataitem_Code','Dataitem_ItemName_ENG')])   
  } else if(c('DataItem_Code') %in% colnames(temp)) {  
    tempi <- unique(temp[,c('DataItem_Code','DataItem_ItemName_ENG')])   
  } else if(c('HealthBehaviour_Code') %in% colnames(temp)) {
    tempi <- unique(temp[,c('HealthBehaviour_Code','HealthBehaviour_ItemName_ENG')])   
  } else if(c('Measure_ItemName_ENG') %in% colnames(temp)) {
    tempi <- unique(temp[,c('Measure_Code','Measure_ItemName_ENG')])
  } else if(c('Month_ItemName_ENG') %in% colnames(temp)) {
    tempi <- unique(temp[,c('Month_Code','Month_ItemName_ENG')])
  } else if(c('Intouch_ItemName_ENG') %in% colnames(temp)) {
    tempi <- unique(temp[,c('Intouch_Code','Intouch_ItemName_ENG')])
  } else if(c("Activity_ItemName_ENG") %in% colnames(temp)) {
    tempi <- unique(temp[,c('Activity_Code','Activity_ItemName_ENG')]) # care0138 also stratifies by Accommodation
  } else if(c("Category_ItemName_ENG") %in% colnames(temp)) {
    tempi <- unique(temp[,c('Category_Code','Category_ItemName_ENG')]) # care0147 also stratifies by Gender_ItemName_ENG
  } else if(c("AgeGroup_ItemName_ENG") %in% colnames(temp)) {
    tempi <- unique(temp[,c('AgeGroup_Code','AgeGroup_ItemName_ENG')]) # care0150 also stratifies by ChildStatus and Immunisations 
  } else if(c("Age_ItemName_ENG") %in% colnames(temp)) {
    tempi <- unique(temp[,c('Age_Code','Age_ItemName_ENG')]) # care0150 also stratifies by ChildStatus and Immunisations 
  } else if(c("ChildStatus_ItemName_ENG") %in% colnames(temp)) {
    tempi <- unique(temp[,c('ChildStatus_Code',"ChildStatus_ItemName_ENG")]) # care0151 stratifies by MentalHealth
  } else if(c("Need_ItemName_ENG") %in% colnames(temp)) {
    tempi <- unique(temp[,c('Need_Code',"Need_ItemName_ENG")]) 
  } else if(c("Needforcare_ItemName_ENG") %in% colnames(temp)) {
    tempi <- unique(temp[,c('Needforcare_Code',"Needforcare_ItemName_ENG")])         
  } else if(c("Adoptions_ItemName_ENG") %in% colnames(temp)) {
    tempi <- unique(temp[,c('Adoptions_Code','Adoptions_ItemName_ENG')]) # care0044 also stratifies for Duration
  } else if(c("Placement_ItemName_ENG") %in% colnames(temp)) {
    tempi <- unique(temp[,c('Placement_Code','Placement_ItemName_ENG')]) 
  } else if(c("Source_ItemName_ENG") %in% colnames(temp)) {
    tempi <- unique(temp[,c('Source_Code','Source_ItemName_ENG')])
  } else if(c("Indicator_ItemName_ENG") %in% colnames(temp)) {
    tempi <- unique(temp[,c('Indicator_Code','Indicator_ItemName_ENG')])
  } else if(c("Notification_ItemName_ENG") %in% colnames(temp)) {
    tempi <- unique(temp[,c('Notification_Code','Notification_ItemName_ENG')])
  } else if(c("CauseofDeath_ItemName_ENG") %in% colnames(temp)) {
    tempi <- unique(temp[,c('CauseofDeath_Code','CauseofDeath_ItemName_ENG')])
  } else if(c("Careprovided_ItemName_ENG") %in% colnames(temp)) {
    tempi <- unique(temp[,c('Careprovided_Code','Careprovided_ItemName_ENG')])
  } else if(c('Date_Code') %in% colnames(temp)) {
    tempi <- unique(temp[,c('Date_Code','Date_ItemName_ENG')])   
  }
  
  # rename the different columns into a single identity
  colnames(tempi)[grepl('ItemName_ENG',colnames(tempi))]  <- 'itemname'  
  colnames(tempi)[grepl('_Code',colnames(tempi))]  <- 'code'
  tempi$code <- paste0(tempi$code)

  tempi$Dataset <- paste0(i)
  tempi$Description <- unique(health_metadata$Description_ENG[health_metadata$Dataset == paste0(i)])
  chmetashort <- bind_rows(indicators_data, tempi)
  rm(tempi,temp)
}

# to test that we have no duplicates for key variables
table(duplicated(indicators_data[,c('Localauthority_Code','Localauthority_ItemName_ENG',
                           'Dataitem_Code','Dataitem_ItemName_ENG',
                           'Dataset','Description_ENG','category','Year_Code')]))

# we group to keys and aggregate to the columns of interest in terms of modelling
chdata2 <- indicators_data %>% group_by(Localauthority_Code,Localauthority_ItemName_ENG,
                               Dataitem_Code,Dataitem_ItemName_ENG,
                               Dataset,category) %>% 
  # value -999 dropped as missing
  filter(Data != -999) %>%
  
  # create improvement column which codes positive changes as 1 and else as 0
  mutate(improvement = ifelse((Data - lag(Data,
                                          default = first(Data), order_by = Year_Code)) > 0,1,0),
  # create a denominator column for N - 1 rows
         denominator = n() - 1,
  # initial and final values for each indicator
         initial_value = first(Data), 
  # initial value for the indicator
          final_value = last(Data)) %>%

  # we drop indicators with 4 or fewer observed values
  filter(denominator > 4) %>%
  
  # create success rate which considers improvement spells over total spells
  mutate(success_rate = sum(improvement, na.rm = T) / denominator) %>%
  
  # we rename local authority columns to match budget data
  rename('Authority_ItemName_ENG' = 'Localauthority_ItemName_ENG',
         'Authority_Code' = 'Localauthority_Code') %>% 

  # create min and max values for each indicators based on the data available
    group_by(Dataitem_Code,Dataitem_ItemName_ENG) %>% 
      mutate(min_value = min(Data, na.rm = T), max_value = max(Data, na.rm = T)) %>%
  
  # select and deduplicate
  select(Authority_ItemName_ENG,Authority_Code,Dataitem_Code,Dataitem_ItemName_ENG,
         Dataset,category, denominator, initial_value, final_value, success_rate, 
         max_value, min_value) %>% 
          unique()

######################################
## 2a. Retrieve spending data
## 2b. Aggregate for analysis on ppi
######################################

# we export codes and description 
write.csv(statswales_get_dataset("lgfs0016") %>% filter(Column_ItemName_ENG == "Net (current) expenditure") %>%
            select(Row_ItemName_ENG,Row_Code) %>% unique(), paste0(outputpath,"budget_description.csv"), row.names = F)

# to retrieve data
spend_data <- statswales_get_dataset("lgfs0016") %>% 

  # Only focusing on Net expenditure for the moment
  filter(Column_ItemName_ENG == "Net (current) expenditure") %>%
 
  # We get the average spending through the years 
  group_by(Authority_ItemName_ENG,Authority_Code,Row_ItemName_ENG,
                  Row_Code) %>% summarise(expenditure = mean(Data),
                                          initial_year = min(Year_Code),
                                          final_year = max(Year_Code))

############################################################
# 3. Temporarily join expenditure data with indicators 
# Natural language analysis produces the relational table
# which replaces this section

matrix <- spend_data %>% inner_join(chdata2)
write.csv(matrix, paste0(outputpath,"matrix.csv"), row.names = F)
