var AInterface = function(Engin){
    this.Outputs = [];
	this.AIoutput = [];
    this.Engine = Engin;
    this.addLimbOutput = function(name,callback,data) {
        this.Outputs.push({NumberOfValues:3,Name:name,Type:"Limb/Rotation",CallBack:callback,Rotation:new this.Engine.Quaternion(),Smoothing:0.5,Data:data});
    }
    this.addSingleValueOutput = function(name,callback,data) {
        this.Outputs.push({NumberOfValues:1,Name:name,Type:"SingleValue/Number",CallBack:callback,Smoothing:0.5,Value:0,Data:data});
    }
    this.addBooleanOutput = function(name,callback,data) {
        this.Outputs.push({NumberOfValues:1,Name:name,Type:"Boolean/Bit",CallBack:callback,Smoothing:0.5,Value:0,Data:data});
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
    this.FT = function(timeDomain,len,m) {
        var freqDomain = [];
        m = m || 1;
        for (var i = 0; i < len; i++) {
            var x = 0;
            var y = 0;
            var r = i*m*2*Math.PI/timeDomain.length;
            for (var j = 0; j < timeDomain.length; j++) {
                x += timeDomain[j]*Math.cos(j*r);
                y += timeDomain[j]*Math.sin(j*r);
            }
            // freqDomain.push([x,y]);
            freqDomain.push(Math.hypot(x,y)/timeDomain.length);
        }
        return freqDomain;
    }
    this.cycle = function(img,otherinp) {
        if (this.AI && this.Engine) {
            var res = this.AI.cycle(img,otherinp,true);
			this.AIoutput = res;
            var result = {RawOutput:res};
            var idx = 0;
            for (var i=0; i<this.Outputs.length; i++) {
                if (this.Outputs[i].Type == "Limb/Rotation") {
                    var arr = [res[idx],res[idx+1],res[idx+2]];
                    // arr[0]--;
                    // arr[1]--;
                    // arr[2]--;
					//var arr = [Math.cos(res[idx]),Math.cos(res[idx+1]),Math.cos(res[idx+2])];
					//var arr = [Math.sin(res[idx]),Math.sin(res[idx+1]),Math.sin(res[idx+2])];
                    var d = Math.hypot(arr[0],arr[1],arr[2]);
                    if (d > 1) {
                        arr[0] /= d;
                        arr[1] /= d;
                        arr[2] /= d;
                        d = 1;
                    }
                    var q = new this.Engine.Quaternion([arr[0],arr[1],arr[2],Math.sqrt(1-(d*d))]).Normalize();
                    if (q.Dot(this.Outputs[i].Rotation) < 0) {
                        q.value = [-q.value[0],-q.value[1],-q.value[2],-q.value[3]];
                    }
                    this.Outputs[i].Rotation.SlerpThisBy(q,1-this.Outputs[i].Smoothing);
                    result[this.Outputs[i].Name] = this.Outputs[i].Rotation;
                    if (this.Outputs[i].CallBack) {
                        this.Outputs[i].CallBack(result[this.Outputs[i].Name],this.Outputs[i].Data);
                    }
                } else if (this.Outputs[i].Type == "FFT/Audio") {
                    result[this.Outputs[i].Name] = res.slice(idx,idx+this.Outputs[i].NumberOfValues);
					for (var q=0; q<result[this.Outputs[i].Name].length; q++) {
						result[this.Outputs[i].Name][q] = Math.max(result[this.Outputs[i].Name][q],0);
					}
                    if (this.Outputs[i].CallBack) {
                        this.Outputs[i].CallBack(result[this.Outputs[i].Name],this.Outputs[i].Data);
                    }
                } else if (this.Outputs[i].Type == "SingleValue/Number") {
                    result[this.Outputs[i].Name] = ((1-this.Outputs[i].Smoothing)*res[idx])+(this.Outputs[i].Smoothing*this.Outputs[i].Value);
                    this.Outputs[i].Value = result[this.Outputs[i].Name];
                    if (this.Outputs[i].CallBack) {
                        this.Outputs[i].CallBack(result[this.Outputs[i].Name],this.Outputs[i].Data);
                    }
                }
                else if (this.Outputs[i].Type == "Boolean/Bit") {
                    result[this.Outputs[i].Name] = ((1-this.Outputs[i].Smoothing)*res[idx])+(this.Outputs[i].Smoothing*this.Outputs[i].Value);
                    this.Outputs[i].Value = result[this.Outputs[i].Name];
                    if (this.Outputs[i].CallBack) {
                        this.Outputs[i].CallBack(result[this.Outputs[i].Name] > 0,this.Outputs[i].Data);
                    }
                }
                idx += this.Outputs[i].NumberOfValues;
            }
            return result;
        }
    }
    this.train = function(a,b) {
        if (this.AI) {
            var outp = new Float32Array(this.AI.rnn.outputSize);
            var idx = 0;
            for (var i=0; i<this.Outputs.length; i++) {
                var val = a[this.Outputs[i].Name];
                if (this.Outputs[i].Type == "Limb/Rotation") {
                    var v = [val.value[0],val.value[1],val.value[2]];
                    if (val.value[3] < 0) {
                        v = [-v[0],-v[1],-v[2]];
                    }
                    // v[0]++;
                    // v[1]++;
                    // v[2]++;
					//v = [Math.acos(v[0]),Math.acos(v[1]),Math.acos(v[2])];
                    outp.set([v[0],v[1],v[2]],idx);
                    idx += 3;
                } else if (this.Outputs[i].Type == "FFT/Audio") {
                    outp.set(val,idx);
                    idx += this.Outputs[i].NumberOfValues;
                }
            }
            return this.AI.trainCurrent(outp,b);
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