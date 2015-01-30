# Script test

outFile = "./saved/save_data.data"

def save((W_size,b,c,W),outFile):
    
    # Save parameters to file and create directory if it doesn't exist
    # Check if output file exists
    if outFile[-1] == '/':
        outFile = outFile[:-1]
    
    baseName = os.path.basename(outFile)
    dirName = os.path.dirname(outFile)
    
    # If not then create file
    if not os.path.isdir(dirName):
        os.makedirs(dirName)
    
    print >> "Hello World!"
 
    f.close()
