var AInterface = function(Engin){
    this.Outputs = [];
    this.Engine = Engin;
    this.addLimbOutput = function(name,callback,data) {
        this.Outputs.push({NumberOfValues:3,Name:name,Type:"Limb/Rotation",CallBack:callback,Rotation:new this.Engine.Quaternion(),Smoothing:0.9,Data:data});
    }
    this.addAudioOutput = function(name,callback,data) {
        this.Outputs.push({NumberOfValues:384,Name:name,Type:"FFT/Audio",CallBack:callback,Data:data});
    }
    this.InitalizeAI = function(opts) {
        if (ReconstructionAI) {
            var o = {
                Vision:{
                    VisionResolution:opts.InputResolution,
                    layerKernelSizes:[],
                    CanTrain:opts.Trainable
                },
                Brain:{
                    layers:[],
                    ExtraInputs:opts.OtherInputs,
                    CanTrain:opts.Trainable
                },
                Float16:opts.Float16
            };
            for (var i=0; i<opts.Layers.length; i++) {
                if (opts.Layers[i] instanceof Array) {
                    o.Vision.layerKernelSizes.push(opts.Layers[i]);
                } else {
                    o.Brain.layers.push(opts.Layers[i]);
                }
            }
            var OutputNeurons = 0;
            for (var i=0; i<this.Outputs.length; i++) {
                OutputNeurons += this.Outputs[i].NumberOfValues;
            }
            o.Brain.layers.push(OutputNeurons);
            this.AI = new ReconstructionAI(o);
        } else {
            throw new Error("ReconstructionAI not Available");
        }
    }
    this.ImportAI = function(data) {
        if (this.AI) {
            this.AI.Import(data);
        }
    }
    this.ExportAI = function() {
        if (this.AI) {
            return this.AI.Export();
        }
    }
    this.cycle = function(img,otherinp) {
        if (this.AI && this.Engine) {
            var res = this.AI.cycle(img,otherinp,true);
            var result = {};
            var idx = 0;
            for (var i=0; i<this.Outputs.length; i++) {
                if (this.Outputs[i].Type == "Limb/Rotation") {
                    var arr = [res[idx],res[idx+1],res[idx+2]];
                    var d = Math.hypot(arr[0],arr[1],arr[2]);
                    if (d > 1) {
                        arr[0] /= d;
                        arr[1] /= d;
                        arr[2] /= d;
                        d = 1;
                    }
                    var q = new this.Engine.Quaternion([arr[0],arr[1],arr[2],Math.sqrt(1-(d*d))]).Normalize();
                    this.Outputs[i].Rotation.SlerpThisBy(q,1-this.Outputs[i].Smoothing);
                    result[this.Outputs[i].Name] = this.Outputs[i].Rotation;
                    if (this.Outputs[i].CallBack) {
                        this.Outputs[i].CallBack(result[this.Outputs[i].Name],this.Outputs[i].Data);
                    }
                } else if (this.Outputs[i].Type == "FFT/Audio") {
                    result[this.Outputs[i].Name] = res.slice(idx,idx+this.Outputs[i].NumberOfValues);
                    if (this.Outputs[i].CallBack) {
                        this.Outputs[i].CallBack(result[this.Outputs[i].Name],this.Outputs[i].Data);
                    }
                }
                idx += this.Outputs[i].NumberOfValues;
            }
            return result;
        }
    }
    this.train = function(a,b,c,d) {
        if (this.AI) {
            return this.AI.train(a,b,c,d);
        }
    };
    this.ReinforcementTrain = function(a,b,c,d) {
        if (this.AI) {
            return this.AI.ReinforcementTrain(a,b,c,d);
        }
    };
    this.finishEpoch = function(a,b,c,d) {
        if (this.AI) {
            return this.AI.finishEpoch(a,b,c,d);
        }
    };
    this.resetZeros = function(a,b,c,d) {
        if (this.AI) {
            return this.AI.resetZeros(a,b,c,d);
        }
    };
}