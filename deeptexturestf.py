'''
    @author: Md Sarfarazul Haque, loads of modifications by Stamatios Matziounis
    This file contains the main class, of this project, 
    that synthesise a texture based on the provided texture, or multiple textures.
    This Project is an implementation of following research paper:
    Citations{
        @online{
            1505.07376,
            Author = {Leon A. Gatys and Alexander S. Ecker and Matthias Bethge},
            Title = {Texture Synthesis Using Convolutional Neural Networks},
            Year = {2015},
            Eprint = {1505.07376},
            Eprinttype = {arXiv},
        }
    }
    NB: I have used fchollet's style transfer project as reference for this project.
    NB: Citations are in BibLaTeX format.
    As mentioned by authors replacing MaxPooling2D layers of VGG19 with AveragePooling2D layers result in 
    good results.
    So to fulfill this I have customized the original VGG19 model written by fchollet by replacing 
    MaxPooling2D layers with AveragePooling2D layers.
'''


''' Importing Packages '''
import numpy as np 
import os
from scipy.optimize import fmin_l_bfgs_b
if os.name == 'nt':
    from keras.preprocessing import image
else:
    import image
import time
#from scipy.misc import imsave
#from keras.preprocessing.image import save_img as imsave
from keras.applications import vgg19
#from matplotlib import pyplot as plt

# This is a customized VGG19 network taken from fchollet implementation of VGG19
# from https://github.com/fchollet/deep-learning-models/blob/master/vgg19.py
from vgg19 import VGG19      
from keras import backend as K 
import tensorflow as tf
tf.compat.v1.disable_eager_execution()



class DeepTexture(object):
    ''' This is the main class that is going to synthesise textures based on given ones '''

    def __init__(self, tex_name, tex_path, gen_prefix='result', base_img_path=None,saveLoc_ = 'results/'):
        ''' Method to initialize variables '''

        '''
            @tex_name: Name that is printed on console
            @tex_path: Path(s) to the `Texture Image`, the image at this location will be used as reference for 
                        texture synthesis
            @gen_prefix: Prefix associated with the output image's names
            @base_img_path: This is the path to an image that can be used as base image to generate texture on.
                            It's default value being `None`. If not `None` it should point to an image that can be
                            used instead of random noise to synthesis texture on
        '''
        # Initializing name and iteration index
        self.name = tex_name
        self.name2 = tex_name #this one is const
        self.total_iterations = 0
        self.val = np.inf
        self.saveLoc = saveLoc_
        # For many images
        self.tex_gram_matrices = []
        self.gen_gram_matrices = {}
        self.f_functions = []
        self.layer_losses = []
        self.layer_loss_scores = []

        # Getting size of the input texture image.
        if(isinstance(tex_path,list)):
            self.width, self.height = image.load_img(tex_path[0]).size    
        else:
            self.width, self.height = image.load_img(tex_path).size

        # Initializing loss value and gradient values as `None`
        self.loss_value = None
        self.grad_values = None
        self.channels = 3

        # To handle the case when base image is `None`
        # This generate a random noise matrix of size of our texture matrix.
        if base_img_path == None:
            x = np.random.rand(self.height, self.width, 3)

            #plt.imshow(x,interpolation='nearest')
            #plt.show()

            # Converting [Width, Height, Channels] to [1, Width, Height, Channels]
            x = np.expand_dims(x, axis=0)

            # Preprocessing the noise image for inferencing through VGG19 model
            self.base_img = vgg19.preprocess_input(x.astype(np.float32))
        else:
            print("Successfully loaded image:",base_img_path)
            self.base_img = self.preprocess_image(base_img_path) # If base_img_path is not `None` then use that image as base image

        # Setting texture image path(s) and prefix values
        self.tex_path = tex_path
        self.gen_prefix = self.name+'_'+gen_prefix
        # Setting the value of input_shape
        if K.image_data_format() == 'channels_last':
            self.input_shape = (1, self.height, self.width, self.channels)
        else:
            self.input_shape = (1, self.channels, self.height, self.width)



    def preprocess_image(self, img_path):
        '''
            This function makes an image ready to be inferentiable by preprocessing it according
            to VGG19 paper.
            @img_path: Path to an image to be preprocessed
            @return: Preprocessed image
        '''

        # Load the image using keras helper class `image`
        img = image.load_img(img_path, target_size=(self.height, self.width))
        # Converting image to array
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Applying preprocessing to the image
        img = vgg19.preprocess_input(img.astype(dtype=np.float32))
        return img


    
    def deprocess_image(self, x):
        '''
            This method deprocess the preprocessed image so that it can be saved to disk as normal images.
            @x: Image matrix
            @return: Converted image
        '''

        # Checking the data format supported by the backend
        if K.image_data_format() == 'channels_first':
            x = x.reshape((self.channels, self.height, self.width))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((self.height, self.width, self.channels))
        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    

    def gram_matrix(self, x):
        '''
            It is a function around which this application is built on. 
            This function calculates the gram matrix of the output of an intermediate layer.
            For more information about gram matrix do give a shot to above mentioned paper.
            @x: Feature map at some intermediate layer of VGG19 model
            @return: Calculated Gram Matrix
        '''

        # To check if input is a tensor or not
        # If not converting that x feature map to a tensor
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)

        # Getting the shape of the tensor
        shape = tf.shape(x)

        # Calculating F matrix by reshaping input tensor
        F = K.reshape(x, [shape[0], shape[1]*shape[2], shape[3]])

        # Calculating and preprocessing G, the Gram Matrix associated with a layer
        gram = tf.matmul(F, F, adjoint_a=True)
        gram /= 2*tf.cast(shape[1]*shape[2],dtype=tf.float32)
        return gram



    def get_loss_per_layer(self, tex, gen):
        '''
            This function calculates the loss associated with a particular layer's output
            @tex: Layer's feature map coming from texture image
            @gen: Layer's feature map coming from synthesised image
            @return: Calculated loss assiciated with the current layer.
        '''

        # Get Gram Matrix of tex feature map
        Tex = self.gram_matrix(tex)
        # Get Gram Matrix of synthesised feature map
        Gen = self.gram_matrix(gen)
        return K.sum(K.square(tf.subtract(Tex, Gen)))


    def eval_loss_and_grads(self, x, loss_function):
        '''
            This function calculates the total loss associated with synthesised with respect to the texture image.
            This function also calculates the total gradient of total loss with respect to the synthesised image. 
            @x: Current intermediate synthesised image
            @return loss_value, grad_values
                @loss_value: Total loss associated with the intermediate sysnthesised 
                            image with respect to texture image
                @grad_values: Total gradient of the loss function with respect to intermediate
                            synthesised image.
        '''

        # Checking and reshaping the 
        if K.image_data_format() == 'channels_first':
            x = x.reshape((1, 3, self.height, self.width))
        else:
            x = x.reshape((1, self.height, self.width, 3))

        # Getting loss_value and grad_values in the form of a list.
        outs = loss_function([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values



    def get_loss(self, x):
        '''
            This is a helper function to help optimizer to get loss function.
            @x: Input intermediate synthesised image.
            @return: Loss function to be feeded to the optimizer.
        '''
        assert self.loss_value is None
        # Getting loss and grad values.
        loss_value, grad_values = self.eval_loss_and_grads(x,self.f_outputs)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return loss_value

    def get_grads(self, x):
        '''
            This is a helper function to help optimizer to get gradient values.
            @x: Input intermediate synthesised image.
            @return: Gradient values to be feeded to the optimizer. 
        '''
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
        


    def buildFunctions(self,features = 'all',lossIndices=None,withLoss=0,varLoss = 1):
        '''
            This is a helper function of this class that wraps everything related to the functionality
            of this project into it, except the iteration running.
            @features: VGG19 layers to be selected for texture synthesisation, default being `all` taking 
                        every layer into consideration for synthesisation of the texture.
                        Other options being `pool` taking outputs only from pooling layers or a set of layers
                        named individually
            
            @lossIndices: A dictionary which contains the index of a texture for every layer taken into account. key = layer, value = texture-index
            @withLoss: A parameter that determines whether a texture will compute the losses for each seperate layer by running iterations. =1 for loss, =0 for normal use
        '''

        # Creating variables and placeholders
        #if(isinstance(self.tex_path,list) and lossIndices == None):
        #   raise ValueError("Error: You didn't provide any indices to determine which texture's layer's loss to use.")
        if(isinstance(self.tex_path,list)):
            print("Notice: Multiple textures detected, preprocessing images")
            tex_img = []
            for i in self.tex_path:
                tex_img.append(K.variable(self.preprocess_image(i)))
        else:
            tex_img = K.variable(self.preprocess_image(self.tex_path))
        self.gen_img = K.placeholder(shape=self.input_shape)

        
        # Creating input_tensor(s) by concatenating the two tensors
        if(isinstance(tex_img,list)):
            input_tensor = []
            for i in tex_img:
                input_tensor.append(K.concatenate([i, self.gen_img], axis=0))
                self.tex_gram_matrices.append({})
                self.layer_losses.append({})
                self.f_functions.append({})
                self.layer_loss_scores.append({})
        else:
            input_tensor = K.concatenate([tex_img, self.gen_img], axis=0)

        # Getting model(s)
        if(isinstance(input_tensor,list)):
            model = []
            for i in input_tensor:
                model.append(VGG19(include_top=False, input_tensor=i, weights='imagenet'))
        else:
            model = VGG19(include_top=False, input_tensor=input_tensor, weights='imagenet')

        # Creating output dictionary for the model.
        if(isinstance(model,list)):
            outputs_dict = []
            for i in model:
                outputs_dict.append(dict([(layer.name, layer.output) for layer in i.layers]))

        else:
            outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

        # Initializing loss variable
        loss = K.variable(0.)

        # Creating flag variable for future use
        flag = True

        # Getting names of all the layers in the model
        if (isinstance(model,list)):
            all_layers = [layer.name for layer in model[0].layers[1:]]
        else:
            all_layers = [layer.name for layer in model.layers[1:]]

        # Checking which layers to be used for reference
        if features == 'all':
            self.feature_layers = all_layers
        elif features == 'pool':
            self.feature_layers = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']
        elif isinstance(features, (list, tuple)):
            for f in features:
                if f not in all_layers:
                    flag = False
            if flag:
                self.feature_layers = features
            else:
                raise ValueError('`features` should be either `all` or `pool` or a set of names from layer.name from model.layers')
        else:
            raise ValueError('`features` should be either `all` or `pool` or a set of names from layer.name from model.layers')


        for layer_name in self.feature_layers:
            gen_features = outputs_dict[0][layer_name][1,:,:,:]
            self.gen_gram_matrices[layer_name] = tf.expand_dims(gen_features, axis=0)

        # Initializing x with base image.    
        self.xx = self.base_img

        # Getting features for Texture Image as well as Synthesised Image as per model
        for layer_name in self.feature_layers:
            for i in range(len(outputs_dict)):
                layer_features = outputs_dict[i][layer_name]
                tex_features = layer_features[0, :, :, :]
                tex_features = tf.expand_dims(tex_features, axis=0)
                self.tex_gram_matrices[i][layer_name] = self.gram_matrix(tex_features)
                # Getting loss per layer for each texture
                layer_loss = self.get_loss_per_layer(tex_features, self.gen_gram_matrices[layer_name])
                self.layer_losses[i][layer_name] = layer_loss
                
                grads = tf.gradients(layer_loss, self.gen_img)
                
                outputs = [layer_loss]
                if isinstance(grads, (list, tuple)):
                    outputs += grads
                else:
                    outputs.append(grads)
                self.f_functions[i][layer_name] = K.function([self.gen_img],outputs)
                self.layer_loss_scores[i][layer_name] = self.eval_loss_and_grads(self.xx,self.f_functions[i][layer_name])[0]
                
    def UpdateTex(self,avg = 0,accent = None):
        loss = K.variable(0.)
        for layer_name in self.feature_layers:
            tempscore = -1
            minIndex = -1
            for i in range(len(self.layer_loss_scores)):
                print(layer_name)
                if(tempscore == -1 or tempscore > self.layer_loss_scores[i][layer_name]):
                    minIndex = i
                    tempscore = self.layer_loss_scores[i][layer_name]
                    
                if(avg == 1):
                    if(layer_name == 'block_pool2' or layer_name == "block_pool3"):
                        loss = loss + K.variable(0.8) * self.layer_losses[i][layer_name]
                    elif(layer_name == 'block_conv4'):
                        loss = loss + K.variable(1.2) * self.layer_losses[minIndex][layer_name]
                    else:
                        loss = loss + self.layer_losses[i][layer_name]
            #print("Min Index for layer %s is %d"%(layer_name,minIndex))
            if (avg == 0):
                if(layer_name == 'block_pool2' or layer_name == "block_pool3"):
                    loss = loss + K.variable(0.8) * self.layer_losses[minIndex][layer_name]
                elif(layer_name == 'block_conv4'):
                    loss = loss + K.variable(1.2) * self.layer_losses[minIndex][layer_name]
                else:
                    loss = loss + self.layer_losses[minIndex][layer_name]
        
        grads = tf.gradients(loss, self.gen_img)
        
        outputs = [loss]
        if isinstance(grads, (list, tuple)):
            outputs += grads
        else:
            outputs.append(grads)
        self.f_outputs = K.function([self.gen_img],outputs)


    def Update(self):
        for layer_name in self.feature_layers:
            for i in range(len(self.layer_loss_scores)):
                self.layer_loss_scores[i][layer_name] = self.eval_loss_and_grads(self.xx,self.f_functions[i][layer_name])[0]


    def sv_img(self,iteration,):
        '''
            This is a helper function of this class that exports the image to a file with its iteration in the filename

            
            @iteration: Number printed at filename
        '''
        # Deprocessing image
        img = self.deprocess_image(self.xx.copy())
        self.fname = 'data/' + self.saveLoc + self.gen_prefix + '_at_iteration_%d.png' % iteration

        # Saving the synthesised image
        image.save_img(self.fname, img)
        print('Image saved as', self.fname)



    def runIterations(self, iterations=50,countIterations=1, printInterval=100, save=1,update = 0):
        '''
            This is the main function of this class that runs the iterations of the fmin_l_bfgs_b algorithm, then exports the result

            
            @iteration: Number printed at filename
            @countIterations: Default value 1, if 0 the training will not count the iterations for the filenames. Used for score gathering
            @printInterval: The interval which a small report is printed
            @save: A parameter that determines whether the iteration is going to return an image or more. 1 = save image at the end, >1 = save in intervals
        '''
        # Reducing total loss using fmin_l_bfgs_b function from scipy.
        # For more information regarding fmin_l_bfgs_b refer to https://www.google.com

        # QA said this should be contained in the code.
        if(iterations<=0):
            raise ValueError("You have provided an invalid amount of iterations. 'iterations' must be a positive integer.")
        if(printInterval>iterations):
            if(iterations/2>0):
                printInterval = int(iterations/2)
            else:
                printInterval = int(iterations)
        
        # The training cycle starts here
        if(printInterval>0):
            print("Starting %d iterations with %d print interval" %(iterations,printInterval))
        start_time = time.time()
        for i in range(iterations+1):

            # Evalutaing for one iteration.
            self.xx, min_val, info = fmin_l_bfgs_b(func=self.get_loss, x0=self.xx.flatten(), fprime=self.get_grads, maxfun=10)
            
            # Using Save value as interval of saving
            #self.sv_img(self.total_iterations+i)
            if ( save>1 and (((i+1) % save) == 1)):
                self.sv_img(self.total_iterations+i) 
            
            if (update > 0 and (i+1)%update==1):
                self.Update()
                self.UpdateTex()
            
            if (printInterval>0 and (i+1)%printInterval==1):
                print('%s: Current loss at iteration %d is: %d' %(self.name,self.total_iterations+i,min_val))
                if (self.val<=1.00001*min_val):  
                    if(min_val >= self.val):
                        print("New value not better than previous best. Stopping")
                    else:
                        print("New value less than 0.001% better than the previous. Stopping")
                    iterations = i
                    
                    # *TODO* CHANGE THIS TO NORMAL
                    #self.val=min_val
                    break
                else:
                    self.val=min_val
            # print(info)

        # Getting ending time
        end_time = time.time()

        # Updating total iterations
        if(countIterations==1):
            self.total_iterations+=iterations

        # Saving the generated image. The not part contains the case: "the last image is saved twice" 
        if(save > 0 and not(save>1 and (((iterations+1) % save) == 1))):
            self.sv_img(self.total_iterations)

        print('%s: %d iterations completed in %ds' % (self.name,iterations, end_time - start_time))
        returnlist = [end_time-start_time,min_val]
        return returnlist
def getInput():
    '''
            This is a helper function that gets a positive input of an integer recursively for the program to run
    '''
    iterations = input()
    
    try:
        iterations = int(iterations)
        if (iterations < 0):
            raise ValueError()
    except:
        print("Invalid number, Reminder: The value must be a positive integer, try again: ",end = ' ')
        iterations = getInput()
    
    return iterations
    
if __name__ == "__main__":
    # Sample run.
    #tex_list = ['data/inputs/ruins1/tex_ruins1.png','data/inputs/ruins1/tex_ruins3.png','data/inputs/ruins1/tex_ruins4.png']
    #base_img = 'data/inputs/ruins1/base_ruins22.png'
    #tex = DeepTexture('tex1',tex_list,base_img_path=base_img)
    tex_list = ['data/inputs/ruins4/tex_ruins1.png','data/inputs/ruins4/tex_ruins3.png','data/inputs/ruins4/tex_ruins6.png','data/inputs/ruins4/tex_ruinss1.png','data/inputs/ruins4/tex_ruinss3.png','data/inputs/ruins4/tex_ruinss4.png']
    base_img = 'data/inputs/ruins4/tex_ruinss2-damaged.png'
    tex = DeepTexture('tex1',tex_list,base_img_path=base_img)
    #tex_list = ['data/inputs/pebbles3/pebbles1.png','data/inputs/pebbles3/pebbles2.png','data/inputs/pebbles3/pebbles3.png']
    #base_img = 'data/inputs/pebbles3/pebbles_base.png'
    #tex = DeepTexture('tex1',tex_list)
    tex.buildFunctions(features='all')
    #count = 1
    tex.UpdateTex(avg=0)
    tex.runIterations(iterations = 10000 ,printInterval = 50,save=200,update = 0)
    #layers = tex.feature_layers
    #
    # Determine the layers that should be used via running iterations on individual layers:
    #
    #for i in layers:
    #    tex.UpdateTex(avg = 1, accent = i)
    #    name = 'tex'+str(count)
    #iterations = getInput()
    #    tex.runIterations(iterations = 10000 ,printInterval = 100,save=200,update = 0)
    #    count+=1
    #    tex = DeepTexture(name,tex_list)
    #    tex.buildFunctions(features='pool')
      
    #a = tex.runIterations()
    
    #tex.xx = tex.preprocess_image("data/inputs/base_ruins222.png")
    #tex.sv_img(-12)

    #print("for tex, loss:",a)
