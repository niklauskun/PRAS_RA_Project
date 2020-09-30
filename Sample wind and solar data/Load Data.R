setwd("C:/Users/wenmi/Desktop/Sample wind and solar data")
BusMapping <- read.csv("Bus Mapping Extra Data.csv")
Wind <- read.csv("Wind_2008 (one week).csv", check.names = F)
Fixed <- read.csv("Fixed_2008 (one week).csv", check.names = F)
Rooftop <- read.csv("Rooftop_2008 (one week).csv", check.names = F)
Tracking <- read.csv("Tracking_2008 (one week).csv", check.names = F)


Wind <- t(Wind)
colnames(Wind) <- Wind[1,]
Wind <- Wind[-1,]
Name <- rownames(Wind)
Wind <- cbind(Name,Wind)
Wind <- merge(Wind,BusMapping, by="Name")
write.csv(Wind,file="C:/Users/wenmi/Desktop/Sample wind and solar data/Wind.csv")

Fixed <- t(Fixed)
colnames(Fixed) <- Fixed[1,]
Fixed <- Fixed[-1,]
Name <- rownames(Fixed)
Fixed <- cbind(Name,Fixed)
Fixed <- merge(Fixed,BusMapping, by="Name")
write.csv(Wind,file="C:/Users/wenmi/Desktop/Sample wind and solar data/Fixed.csv")

Rooftop <- t(Rooftop)
colnames(Rooftop) <- Rooftop[1,]
Rooftop <- Rooftop[-1,]
Rooftop <- rownames(Rooftop)
Rooftop <- cbind(Name,Rooftop)
Rooftop <- merge(Rooftop,BusMapping, by="Name")
write.csv(Wind,file="C:/Users/wenmi/Desktop/Sample wind and solar data/Rooftop.csv")

Tracking <- t(Tracking)
colnames(Tracking) <- Tracking[1,]
Tracking <- Tracking[-1,]
Tracking <- rownames(Tracking)
Tracking <- cbind(Name,Tracking)
Tracking <- merge(Tracking,BusMapping, by="Name")
write.csv(Wind,file="C:/Users/wenmi/Desktop/Sample wind and solar data/Tracking.csv")
