# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:45:09 2018

@author: Mostafa
"""
#%links

#%library


# functions


#%% 
def javaupload():
    """
    # =============================================================================
    #     javaupload()
    # =============================================================================
    this function takes no argument and return a script (string) of java code to
    be used in upload buttons 
    """
    javaupload_code="""
    //function 1 
    function read_file(filename) 
        {
        var reader = new FileReader();
        reader.onload = load_handler;
        reader.onerror = error_handler;
        // readAsDataURL represents the file's data as a base64 encoded string
        reader.readAsDataURL(filename);
        }
    
    // to load the file
    function load_handler(event) 
        {
        var b64string = event.target.result;
        file_source.data = {'file_contents' : [b64string], 'file_name':[input.files[0].name]};
        file_source.trigger("change");
        }
        
    // incase of error
    function error_handler(evt) 
        {
            if(evt.target.error.name == "NotReadableError") 
                {
                alert("Can't read file!");
                }
        }
    
    var input = document.createElement('input');
    
    input.setAttribute('type', 'file');
    
    input.onchange = function()
        {
        if (window.FileReader) 
        {
            read_file(input.files[0]);
        } else {
            alert('FileReader is not supported in this browser');
        }
        }
        
    input.click();
    """
    return javaupload_code

def upload_stations():
    """
    # =============================================================================
    #     javaupload()
    # =============================================================================
    this function takes no argument and return a script (string) of java code to
    be used in upload buttons 
    """
    javaupload_code="""
    //function 1 
    function read_file(filename) 
        {
        var reader = new FileReader();
        reader.onload = load_handler;
        reader.onerror = error_handler;
        // readAsDataURL represents the file's data as a base64 encoded string
        reader.readAsDataURL(filename);
        }
    
    // to load the file
    function load_handler(event) 
        {
        var b64string = event.target.result;
        file_source.data = {'file_contents' : [b64string], 'file_name':[input.files[0].name],
                            'name':[0],'lat':[0],'long':[0]};
        file_source.trigger("change");
        }
        
    // incase of error
    function error_handler(evt) 
        {
            if(evt.target.error.name == "NotReadableError") 
                {
                alert("Can't read file!");
                }
        }
    
    var input = document.createElement('input');
    
    input.setAttribute('type', 'file');
    
    input.onchange = function()
        {
        if (window.FileReader) 
        {
            read_file(input.files[0]);
        } else {
            alert('FileReader is not supported in this browser');
        }
        }
        
    input.click();
    """
    return javaupload_code


def raster_upload():
    rasteruploader_code="""
    //function 1 
    function read_file(filename) 
        {
        var reader = new FileReader();
        
        reader.onload = load_handler;
        reader.onerror = error_handler;
        // readAsDataURL represents the file's data as a base64 encoded string
        reader.readAsDataURL(filename);
        }
    
    // to load the file
    function load_handler(event) 
        {
        //var b64string = event.target.result;
        //file_source.data = {'file_contents' : [b64string], 'file_name':[input.files[0].name]};
        //file_source.trigger("change");
        
        //var urlpath =  "vardah.tiff"

        var oReq = new XMLHttpRequest();
        
        oReq.open("GET", event, true);
        oReq.responseType = "arraybuffer";
        
        oReq.onload = function(oEvent) 
        {
          var t0 = performance.now();
          var tiff = GeoTIFF.parse(this.response);
          var image = tiff.getImage();
          var data = image.readRasters()[0];
          var tiepoint = image.getTiePoints()[0];
          var pixelScale = image.getFileDirectory().ModelPixelScale;
          var t1 = performance.now();
          console.log("Decoding took " + (t1 - t0) + " milliseconds.")
        };
        
        oReq.send(); //start process
    
        }
    
    
    // incase of error
    function error_handler(evt) 
        {
            if(evt.target.error.name == "NotReadableError") 
                {
                alert("Can't read file!");
                }
        }
    
    
    var input = document.createElement('input');
    
    input.setAttribute('type', 'file');
    
    input.onchange = function()
        {
        if (window.FileReader) 
        {
            read_file(input.files[0]);
        } else {
            alert('FileReader is not supported in this browser');
        }
        }
    input.click();
        
    """
    return rasteruploader_code

def warning1():
    """
    
    """
    javacode="""
    function myFunction() {
                            alert("I am an alert box!");
                            }   
    """
    
    return javacode