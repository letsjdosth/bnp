library("beyondWhittle")
?gibbs_np
data = read.csv("./data/ar2_testdata1.csv",header=FALSE)
data = as.numeric(t(data))
data = data - mean(data)

# Start the clock!
ptm <- proc.time()
res <- gibbs_np(data, 5000, 0)
print(proc.time() - ptm)
print(res)
plot(res)
