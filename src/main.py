from tf_unet import unet, util, image_util

#preparing data loading
#data_provider = image_util.ImageDataProvider("Images/Few/train/*.tif")
data_provider = image_util.ImageDataProvider("Images/train/*.tif")
print " 1 YES"
#path where to store the checkpoints
output_path = "results"
print " 2 YES"
#setup & training
# We only have 1 channel as the images are in grey scale
# We have 2 classes (0 for belong to the lung or 1 to doesn't belong to the lung)
#net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2)
net = unet.Unet(layers=3, features_root=32, channels=1, n_class=2)
print " 3 YES" 
trainer = unet.Trainer(net, optimizer="adam")
print " 4 YES"
#path = trainer.train(data_provider, output_path, training_iters=32, epochs=100, write_graph=True)
path = trainer.train(data_provider,output_path, training_iters=2, epochs=1, write_graph=True)
print " Training is completed "
