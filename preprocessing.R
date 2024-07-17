####wave to audio
###preprocessing for next step
setwd("Your/working/directory/")
library(tuneR)
sample_rate=800

####normal TMJ wave data
dat<-read.table("normal.csv",header = T,sep = ",",stringsAsFactors = F)
sampleid<-unique(dat$sampleid)
for (i in 1:length(sampleid)) {
  temp<-dat[which(dat$sampleid == sampleid[i]),3]
  temp_norm<-temp/max(abs(temp))
  wave_temp<-Wave(temp_norm, samp.rate = 100,bit=16)
  wave_temp<-wave_temp*32767
  writeWave(wave_temp,file = paste0("data/L/","0_L",sampleid[i],".wav"),extensible=F)
  print(i)
}

for (i in 1:length(sampleid)) {
  temp<-dat[which(dat$sampleid == sampleid[i]),4]
  temp_norm<-temp/max(abs(temp))
  wave_temp<-Wave(temp_norm, samp.rate = 100,bit=16)
  wave_temp<-wave_temp*32767
  writeWave(wave_temp,file = paste0("data/R/","0_R",sampleid[i],".wav"),extensible=F)
  print(i)
}

dat_L<-dat[,c(1:3)]
dat_L$sampleid<-paste0(dat_L$sampleid,"-L")
colnames(dat_L)<-c("sampleid","time","y")
dat_R<-dat[,c(1:2,4)]
dat_R$sampleid<-paste0(dat_R$sampleid,"-R")
colnames(dat_R)<-c("sampleid","time","y")
dat_S<-rbind(dat_L,dat_R)
sampleid_S<-unique(dat_S$sampleid)
for (i in 1:length(sampleid_S)){
  temp<-dat_S[which(dat_S$sampleid == sampleid_S[i]),3]
  temp_norm<-temp/max(abs(temp))
  wave_temp<-Wave(temp_norm, samp.rate = 100,bit=16)
  wave_temp<-wave_temp*32767
  writeWave(wave_temp,file = paste0("data/S/","0_",sampleid_S[i],".wav"),extensible=F)
  print(i)
}

####patient TMD wave data
dat<-read.table("patient.csv",header = T,sep = ",",stringsAsFactors = F)
sampleid<-unique(dat$sampleid)
for (i in 1:length(sampleid)) {
  temp<-dat[which(dat$sampleid == sampleid[i]),3]
  temp_norm<-temp/max(abs(temp))
  wave_temp<-Wave(temp_norm, samp.rate = 100,bit=16)
  wave_temp<-wave_temp*32767
  writeWave(wave_temp,file = paste0("data/L/","1_L",sampleid[i],".wav"),extensible=F)
  print(i)
}

for (i in 1:length(sampleid)) {
  temp<-dat[which(dat$sampleid == sampleid[i]),4]
  temp_norm<-temp/max(abs(temp))
  wave_temp<-Wave(temp_norm, samp.rate = 100,bit=16)
  wave_temp<-wave_temp*32767
  writeWave(wave_temp,file = paste0("data/R/","1_R",sampleid[i],".wav"),extensible=F)
  print(i)
}

dat_L<-dat[,c(1:3)]
dat_L$sampleid<-paste0(dat_L$sampleid,"-L")
colnames(dat_L)<-c("sampleid","time","y")
dat_R<-dat[,c(1:2,4)]
dat_R$sampleid<-paste0(dat_R$sampleid,"-R")
colnames(dat_R)<-c("sampleid","time","y")
dat_S<-rbind(dat_L,dat_R)
sampleid_S<-unique(dat_S$sampleid)
for (i in 1:length(sampleid_S)){
  temp<-dat_S[which(dat_S$sampleid == sampleid_S[i]),3]
  temp_norm<-temp/max(abs(temp))
  wave_temp<-Wave(temp_norm, samp.rate = 100,bit=16)
  wave_temp<-wave_temp*32767
  writeWave(wave_temp,file = paste0("data/S/","1_",sampleid_S[i],".wav"),extensible=F)
  print(i)
}

