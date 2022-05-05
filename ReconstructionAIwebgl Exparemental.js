var ReconstructionAI = function(opts){
    this.opts = opts;
    this.Float16 = opts.Float16;
    this.canvas = document.createElement('canvas');
    this.gl = this.canvas.getContext('webgl2',{antialias: false, alpha: true, depth: false});
    this.gl.getExtension('EXT_color_buffer_float');
    this.gl.getExtension('EXT_color_buffer_half_float');
    var IB = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, IB);
    this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0,1,2,3,2,1]), this.gl.STATIC_DRAW);
    this.vs = this.gl.createShader(this.gl.VERTEX_SHADER);
    this.gl.shaderSource(this.vs,`#version 300 es
    highp vec4 p[4] = vec4[4](vec4(1.0,1.0,0.0,1.0),vec4(-1.0,1.0,0.0,1.0),vec4(1.0,-1.0,0.0,1.0),vec4(-1.0,-1.0,0.0,1.0));
    void main(){
        gl_Position = p[gl_VertexID];
    }
    `);
    this.gl.compileShader(this.vs);
    if (!this.gl.getShaderParameter(this.vs, this.gl.COMPILE_STATUS)){
        console.error(this.gl.getShaderInfoLog(this.vs));
    }
    var self = this;
    if (opts.Float16) {
        self.RGBAF = this.gl.RGBA16F;
        self.RGBF = this.gl.RGB16F;
		self.RGF = this.gl.RG16F;
        self.RF = this.gl.R16F;
        self.FLOAT = this.gl.HALF_FLOAT;
    } else {
        self.RGBAF = this.gl.RGBA32F;
        self.RGBF = this.gl.RGB32F;
		self.RGF = this.gl.RG32F;
        self.RF = this.gl.R32F;
        self.FLOAT = this.gl.FLOAT;
    }
    function createProgram(fsScorce) {
        var fs = self.gl.createShader(self.gl.FRAGMENT_SHADER);
        self.gl.shaderSource(fs,fsScorce);
        self.gl.compileShader(fs);
        var program = self.gl.createProgram();
        self.gl.attachShader(program, self.vs);
        self.gl.attachShader(program, fs);
        self.gl.linkProgram(program);
        if (!self.gl.getProgramParameter(program, self.gl.LINK_STATUS)) {
            console.error('Could not initialise shaders');
            console.error(self.gl.getShaderInfoLog(fs));
        }
        return program;
    }
    function randn() {
        var u = 0, v = 0;
        while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
        while(v === 0) v = Math.random();
        return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    }
    var drawPgrm = createProgram(`#version 300 es
    uniform sampler2D u_image;
    out highp vec4 o_color;
    void main(){
		highp vec4 val = texelFetch(u_image, ivec2(gl_FragCoord.xy-0.5), 0);
        o_color = val;
        // o_color = texelFetch(u_image, ivec2(gl_FragCoord.xy-0.5), 0);
        //o_color = (val/2.0)+0.5;
		// o_color = vec4((val.xyz/2.0)+0.5,1.0);
    }`);
    this.drawTexture = function(tex,res) {
        this.canvas.width = res[0];
        this.canvas.height = res[1];
        this.gl.viewport(0,0,res[0],res[1]);
        this.gl.scissor(0,0,res[0],res[1]);
        this.gl.useProgram(drawPgrm);
        this.gl.activeTexture(this.gl.TEXTURE0);
        this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
        this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
    }
    this.RCNN = function() {
        this.gl = self.gl;
        this.randomWaB = createProgram(`#version 300 es
            uniform highp mat3 uRand;
            uniform highp float uM;
            const highp float PI = 3.1415926535897932384626433832795;
            const highp float Tau = PI * 2.0;
            const highp float epsilon = 0.0000152587;
            highp vec3 seed = vec3(0.0,0.0,0.0);
            out highp vec4 fragColor;
            highp vec3 random() {
                highp uvec3 x = uvec3(floatBitsToUint(seed));
                x = ((x>>8U)^x.yzx)*110351524U;
                x = ((x>>8U)^x.yzx)*110351524U;
                x = ((x>>8U)^x.yzx)*110351524U;
                seed = mod((vec3(x)/3141593.0)*uRand,1.0);
                //seed = mod((vec3(x)/3141593.0)*uRand,2.000000238418579)-1.0;
                return seed;
            }
            highp vec3 randn() {
                highp vec3 r0 = random();
                highp vec3 r1 = random();
                return clamp(sqrt(-2.0*log(r0))*cos(Tau*r1),vec3(-16.0),vec3(16.0));
            }
            void main(){
                seed = gl_FragCoord.xyz*uRand;
                // randn();
                random();
                //fragColor = vec4(random()*uM,1.0);
				fragColor = vec4(randn()*uM,1.0);
            }
        `);
        this.cycleLayer = createProgram(`#version 300 es
            uniform highp sampler2D uWaB[6];
            uniform highp sampler2D uInpPrev;
            uniform highp sampler2D uInp;
            uniform highp ivec2 uKernelSize;
            out highp vec4 Activation[2];
            highp vec3 activationFunction(highp vec3 v){
                // v = exp(-2.0*v);
                // return (1.0-v)/(1.0+v);

                // return tanh(v);

                return max(v,vec3(0));

                // return v/sqrt(1.0+(v*v));
            }
            void main(){
                highp ivec2 ksx = (uKernelSize)/2;
                highp ivec2 p0 = ivec2(gl_FragCoord.xy-0.5);
                highp ivec2 p1 = (p0*uKernelSize);
                highp vec3 val = vec3(0.0);
                for (highp int x=0;x<uKernelSize[0];x++){
                    for (highp int y=0;y<uKernelSize[1];y++){
                        highp ivec2 p = p1+ivec2(x,y);
                        highp vec3 v = texelFetch(uInp,p,0).xyz;
						val += v*mat3(texelFetch(uWaB[0],p,0).xyz,texelFetch(uWaB[1],p,0).xyz,texelFetch(uWaB[2],p,0).xyz);
						//val += mat3(texelFetch(uWaB[0],p,0).xyz,texelFetch(uWaB[1],p,0).xyz,texelFetch(uWaB[2],p,0).xyz)*v;
                        //val.x += dot(v,texelFetch(uWaB[0],p,0).xyz);
                        //val.y += dot(v,texelFetch(uWaB[1],p,0).xyz);
                        //val.z += dot(v,texelFetch(uWaB[2],p,0).xyz);
                        if (x==ksx[0] && y==ksx[1]){
                            val.x += texelFetch(uWaB[3],p,0).y;
                            val.y += texelFetch(uWaB[4],p,0).y;
                            val.z += texelFetch(uWaB[5],p,0).y;
                        } else {
							v = texelFetch(uInpPrev,p0+ivec2(x,y)-ksx,0).xyz;
							val += v*mat3(texelFetch(uWaB[3],p,0).xyz,texelFetch(uWaB[4],p,0).xyz,texelFetch(uWaB[5],p,0).xyz);
							//val += mat3(texelFetch(uWaB[3],p,0).xyz,texelFetch(uWaB[4],p,0).xyz,texelFetch(uWaB[5],p,0).xyz)*v;
                            //val.x += dot(texelFetch(uWaB[3],p,0).xyz,v);
                            //val.y += dot(texelFetch(uWaB[4],p,0).xyz,v);
                            //val.z += dot(texelFetch(uWaB[5],p,0).xyz,v);
                        }
                    }
                }
                val = activationFunction(val);
                Activation[0] = vec4(val,1.0);
                Activation[1] = vec4(val-texelFetch(uInpPrev,p0,0).xyz,1.0);
            }
        `);
		this.backprop = createProgram(`#version 300 es
            uniform highp sampler2D uGradient[2];
            uniform highp sampler2D uInp;
            uniform highp ivec2 uKernelSize;
            uniform highp sampler2D uWaB[3];
            out highp vec4 OutGradient[2];
			highp vec3 activationFunctionDerivitive(highp vec3 v){
                // return 1.0-(v*v);
                return sign(v);
            }
            void main(){
                highp ivec2 ksx = (uKernelSize)/2;
                highp ivec2 p0 = ivec2(gl_FragCoord.xy-0.5);
                highp ivec2 p1 = ivec2(p0/uKernelSize);
                highp vec3 delta = texelFetch(uGradient[0],p1,0).xyz;
                highp vec3 delta2 = texelFetch(uGradient[1],p1,0).xyz;
                highp ivec2 xy = (p0%uKernelSize);
				highp vec3 deriv = activationFunctionDerivitive(texelFetch(uInp,p0,0).xyz);
				highp vec3 w0 = texelFetch(uWaB[0],p0,0).xyz;
				highp vec3 w1 = texelFetch(uWaB[1],p0,0).xyz;
				highp vec3 w2 = texelFetch(uWaB[2],p0,0).xyz;
                OutGradient[0] = vec4(((w0*delta.x)+(w1*delta.y)+(w2*delta.z))*deriv,1.0);
				OutGradient[1] = vec4(((w0*delta2.x)+(w1*delta2.y)+(w2*delta2.z))*deriv,1.0);
            }
        `);
        this.backpropWaB = createProgram(`#version 300 es
            uniform highp sampler2D uActivationsPrev[2];
            uniform highp sampler2D uGradient[2];
            uniform highp sampler2D uInp[2];
            uniform highp ivec2 uKernelSize;
            uniform highp sampler2D uWaBgradient[6];
			uniform bool islast;
            out highp vec4 OutGradient[6];
            void main(){
                highp ivec2 ksx = (uKernelSize)/2;
                highp ivec2 p0 = ivec2(gl_FragCoord.xy-0.5);
                highp ivec2 p1 = ivec2(p0/uKernelSize);
                highp vec3 delta = texelFetch(uGradient[0],p1,0).xyz;
                highp vec3 delta2 = texelFetch(uGradient[1],p1,0).xyz;
                highp ivec2 xy = (p0%uKernelSize);
				highp vec3 v = texelFetch(uInp[0],p0,0).xyz;
                highp vec3 v2 = texelFetch(uInp[1],p0,0).xyz;
				if (islast) {
					v2 = v-v2;
				}
                OutGradient[0] = vec4(texelFetch(uWaBgradient[0],p0,0).xyz+((v*delta.x)+(v2*delta2.x)),1.0);
                OutGradient[1] = vec4(texelFetch(uWaBgradient[1],p0,0).xyz+((v*delta.y)+(v2*delta2.y)),1.0);
                OutGradient[2] = vec4(texelFetch(uWaBgradient[2],p0,0).xyz+((v*delta.z)+(v2*delta2.z)),1.0);
                if (all(equal(xy,ksx))){
                    OutGradient[3] = vec4(texelFetch(uWaBgradient[3],p0,0).xyz+vec3(delta.x),1.0);
                    OutGradient[4] = vec4(texelFetch(uWaBgradient[4],p0,0).xyz+vec3(delta.y),1.0);
                    OutGradient[5] = vec4(texelFetch(uWaBgradient[5],p0,0).xyz+vec3(delta.z),1.0);
                } else {
                    highp ivec2 p2 = p1+xy-ksx;
					v = texelFetch(uActivationsPrev[0],p2,0).xyz;
                    v2 = texelFetch(uActivationsPrev[1],p2,0).xyz;
                    OutGradient[3] = vec4(texelFetch(uWaBgradient[3],p0,0).xyz+((v*delta.x)+(v2*delta2.x)),1.0);
                    OutGradient[4] = vec4(texelFetch(uWaBgradient[4],p0,0).xyz+((v*delta.y)+(v2*delta2.y)),1.0);
                    OutGradient[5] = vec4(texelFetch(uWaBgradient[5],p0,0).xyz+((v*delta.z)+(v2*delta2.z)),1.0);
                }
            }
        `);
        this.addWaB = createProgram(`#version 300 es
            uniform highp sampler2D uWaBcurrent[6];
            uniform highp sampler2D uWaBgradient[6];
            uniform highp float uMlt;
            out highp vec4 OutWaB[6];
            void main(){
                highp ivec2 p = ivec2(gl_FragCoord.xy-0.5);
				OutWaB[0] = clamp(vec4(texelFetch(uWaBcurrent[0],p,0).xyz+(texelFetch(uWaBgradient[0],p,0).xyz*uMlt),1.0),vec4(-16.0),vec4(16.0));
				OutWaB[1] = clamp(vec4(texelFetch(uWaBcurrent[1],p,0).xyz+(texelFetch(uWaBgradient[1],p,0).xyz*uMlt),1.0),vec4(-16.0),vec4(16.0));
				OutWaB[2] = clamp(vec4(texelFetch(uWaBcurrent[2],p,0).xyz+(texelFetch(uWaBgradient[2],p,0).xyz*uMlt),1.0),vec4(-16.0),vec4(16.0));
				
				OutWaB[3] = clamp(vec4(texelFetch(uWaBcurrent[3],p,0).xyz+(texelFetch(uWaBgradient[3],p,0).xyz*uMlt),1.0),vec4(-16.0),vec4(16.0));
				OutWaB[4] = clamp(vec4(texelFetch(uWaBcurrent[4],p,0).xyz+(texelFetch(uWaBgradient[4],p,0).xyz*uMlt),1.0),vec4(-16.0),vec4(16.0));
				OutWaB[5] = clamp(vec4(texelFetch(uWaBcurrent[5],p,0).xyz+(texelFetch(uWaBgradient[5],p,0).xyz*uMlt),1.0),vec4(-16.0),vec4(16.0));
            }
        `);
        this.initalize = function(ops) {
            this.WaB = [];
            this.activations = [[],[]];
            this.WaBdelta = [];
			this.backpropTextures = [];
            this.DataSetSize = 0;
            this.NumberOfParamiters = 0;
            this.frameBuffer = this.gl.createFramebuffer();
            this.frameBufferTexture = this.gl.createTexture();
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER,this.frameBuffer);
            this.gl.bindTexture(this.gl.TEXTURE_2D,this.frameBufferTexture);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_T,this.gl.CLAMP_TO_EDGE);
            this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBAF,ops.VisionResolution[0],ops.VisionResolution[1],0,this.gl.RGBA,self.FLOAT,null);
            this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER,this.gl.COLOR_ATTACHMENT0,this.gl.TEXTURE_2D,this.frameBufferTexture,0);
            this.gl.viewport(0,0,ops.VisionResolution[0],ops.VisionResolution[1]);
            this.gl.scissor(0,0,ops.VisionResolution[0],ops.VisionResolution[1]);
            this.cycleframeBuffer = this.gl.createFramebuffer();
            this.cycleframeBufferTexture0 = this.gl.createTexture();
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER,this.cycleframeBuffer);
            this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0,this.gl.COLOR_ATTACHMENT1]);
            this.gl.bindTexture(this.gl.TEXTURE_2D,this.cycleframeBufferTexture0);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_T,this.gl.CLAMP_TO_EDGE);
            this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBAF,ops.VisionResolution[0]/ops.layerKernelSizes[0][0],ops.VisionResolution[1]/ops.layerKernelSizes[0][1],0,this.gl.RGBA,self.FLOAT,null);
            this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER,this.gl.COLOR_ATTACHMENT0,this.gl.TEXTURE_2D,this.cycleframeBufferTexture0,0);
            this.cycleframeBufferTexture1 = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D,this.cycleframeBufferTexture1);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_T,this.gl.CLAMP_TO_EDGE);
            this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBAF,ops.VisionResolution[0]/ops.layerKernelSizes[0][0],ops.VisionResolution[1]/ops.layerKernelSizes[0][1],0,this.gl.RGBA,self.FLOAT,null);
            this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER,this.gl.COLOR_ATTACHMENT1,this.gl.TEXTURE_2D,this.cycleframeBufferTexture1,0);
            this.WaBframeBuffer = this.gl.createFramebuffer();
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER,this.WaBframeBuffer);
            this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0,this.gl.COLOR_ATTACHMENT1,this.gl.COLOR_ATTACHMENT2,this.gl.COLOR_ATTACHMENT3,this.gl.COLOR_ATTACHMENT4,this.gl.COLOR_ATTACHMENT5]);
			this.backpropFrameBuffer = this.gl.createFramebuffer();
            for (var i=0;i<6;i++){
                this.NumberOfParamiters += ops.VisionResolution[0]*ops.VisionResolution[1]*3;
                var texture = this.gl.createTexture();
                this.gl.bindTexture(this.gl.TEXTURE_2D,texture);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE);
                this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBAF,ops.VisionResolution[0],ops.VisionResolution[1],0,this.gl.RGBA,self.FLOAT,null);
                this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER,this.gl.COLOR_ATTACHMENT0+i,this.gl.TEXTURE_2D,texture,0);
            }
            this.layerResolutions = [ops.VisionResolution];
            this.inputTexture0 = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D,this.inputTexture0);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_T,this.gl.CLAMP_TO_EDGE);
            this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[0][0],this.layerResolutions[0][1],0,this.gl.RGB,self.FLOAT,null);
            this.inputTexture1 = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D,this.inputTexture1);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_T,this.gl.CLAMP_TO_EDGE);
            this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[0][0],this.layerResolutions[0][1],0,this.gl.RGB,self.FLOAT,null);
            this.inputTextures = [this.inputTexture0,this.inputTexture1];
            this.gl.useProgram(this.cycleLayer);
            this.KernelSizeUniform = this.gl.getUniformLocation(this.cycleLayer,"uKernelSize");
            this.gl.uniform1iv(this.gl.getUniformLocation(this.cycleLayer,"uWaB"),[0,1,2,3,4,5]);
            this.gl.uniform1i(this.gl.getUniformLocation(this.cycleLayer,"uInp"),6);
            this.gl.uniform1i(this.gl.getUniformLocation(this.cycleLayer,"uInpPrev"),7);
			
			this.gl.useProgram(this.backprop);
            this.backpropBackKernelSizeUniform = this.gl.getUniformLocation(this.backprop,"uKernelSize");
            this.gl.uniform1iv(this.gl.getUniformLocation(this.backprop,"uWaB"),[0,1,2]);
			this.gl.uniform1iv(this.gl.getUniformLocation(this.backprop,"uGradient"),[3,4]);
            this.gl.uniform1i(this.gl.getUniformLocation(this.backprop,"uInp"),5);
			
            this.gl.useProgram(this.randomWaB);
            var m = this.gl.getUniformLocation(this.randomWaB,"uM");
            var urand = this.gl.getUniformLocation(this.randomWaB,"uRand");
            this.gl.useProgram(this.backpropWaB);
            this.BackpropKernelSizeUniform = this.gl.getUniformLocation(this.backpropWaB,"uKernelSize");
			this.IsLastBackpropUniform = this.gl.getUniformLocation(this.backpropWaB,"islast");
            this.gl.uniform1iv(this.gl.getUniformLocation(this.backpropWaB,"uWaBgradient"),[0,1,2,3,4,5]);
            this.gl.uniform1iv(this.gl.getUniformLocation(this.backpropWaB,"uGradient"),[6,7]);
            this.gl.uniform1iv(this.gl.getUniformLocation(this.backpropWaB,"uActivationsPrev"),[8,9]);
            this.gl.uniform1iv(this.gl.getUniformLocation(this.backpropWaB,"uInp"),[10,11]);
            this.gl.useProgram(this.addWaB);
            this.addWabMult = this.gl.getUniformLocation(this.addWaB,"uMlt");
            this.addWabKernelSize = this.gl.getUniformLocation(this.addWaB,"uKernelSize");
            this.gl.uniform1iv(this.gl.getUniformLocation(this.addWaB,"uWaBgradient"),[0,1,2,3,4,5]);
            this.gl.uniform1iv(this.gl.getUniformLocation(this.addWaB,"uWaBcurrent"),[6,7,8,9,10,11]);
            this.layerKernelSizes = ops.layerKernelSizes;
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER,this.frameBuffer);
            for (let i=0; i<this.layerKernelSizes.length; i++){
                this.layerResolutions.push([(this.layerResolutions[i][0]/this.layerKernelSizes[i][0]),(this.layerResolutions[i][1]/this.layerKernelSizes[i][1])]);
                var C = [];
                var D = [];
                this.gl.useProgram(this.randomWaB);
                //this.gl.uniform1f(m,1/Math.sqrt(this.layerKernelSizes[i][0]*this.layerKernelSizes[i][1]*6));
				this.gl.uniform1f(m,Math.sqrt(1/(this.layerKernelSizes[i][0]*this.layerKernelSizes[i][1]*6)));
                // this.gl.uniform1f(m,1/Math.sqrt(Math.sqrt(this.layerKernelSizes[i][0]*this.layerKernelSizes[i][1]*6)));
                this.gl.viewport(0,0,this.layerResolutions[i][0],this.layerResolutions[i][1]);
                this.gl.scissor(0,0,this.layerResolutions[i][0],this.layerResolutions[i][1]);
                for (let j=0;j<6;j++){
                    this.gl.uniformMatrix3fv(urand,false,new Float32Array(9).map(function(){return (Math.random()-0.5)*20}));
                    C.push(this.gl.createTexture());
                    this.gl.bindTexture(this.gl.TEXTURE_2D,C[j]);
                    this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.NEAREST);
                    this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.NEAREST);
                    this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE);
                    this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_T,this.gl.CLAMP_TO_EDGE);
                    this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[i][0],this.layerResolutions[i][1],0,this.gl.RGB,self.FLOAT,null);
                    this.gl.drawElements(this.gl.TRIANGLES,6,this.gl.UNSIGNED_SHORT,0);
                    this.gl.bindTexture(this.gl.TEXTURE_2D,C[j]);
                    this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i][0],this.layerResolutions[i][1],0);
					if (ops.CanTrain) {
						D.push(this.gl.createTexture());
						this.gl.bindTexture(this.gl.TEXTURE_2D,D[j]);
						this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.NEAREST);
						this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.NEAREST);
						this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE);
						this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_T,this.gl.CLAMP_TO_EDGE);
						this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[i][0],this.layerResolutions[i][1],0,this.gl.RGB,self.FLOAT,null);
					}
                }
				this.backpropTextures.push([this.gl.createTexture(),this.gl.createTexture()]);
				this.gl.bindTexture(this.gl.TEXTURE_2D,this.backpropTextures[i][0]);
				this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.NEAREST);
				this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.NEAREST);
				this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE);
				this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_T,this.gl.CLAMP_TO_EDGE);
				this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBAF,this.layerResolutions[i+1][0],this.layerResolutions[i+1][1],0,this.gl.RGBA,self.FLOAT,null);
				this.gl.bindTexture(this.gl.TEXTURE_2D,this.backpropTextures[i][1]);
				this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.NEAREST);
				this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.NEAREST);
				this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE);
				this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_T,this.gl.CLAMP_TO_EDGE);
				this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBAF,this.layerResolutions[i+1][0],this.layerResolutions[i+1][1],0,this.gl.RGBA,self.FLOAT,null);
                this.activations[0].push([this.gl.createTexture(),this.gl.createTexture()]);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[0][i][0]);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_T,this.gl.CLAMP_TO_EDGE);
                this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[i+1][0],this.layerResolutions[i+1][1],0,this.gl.RGB,self.FLOAT,null);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[0][i][1]);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_T,this.gl.CLAMP_TO_EDGE);
                this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[i+1][0],this.layerResolutions[i+1][1],0,this.gl.RGB,self.FLOAT,null);
                this.activations[1].push([this.gl.createTexture(),this.gl.createTexture()]);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[1][i][0]);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_T,this.gl.CLAMP_TO_EDGE);
                this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[i+1][0],this.layerResolutions[i+1][1],0,this.gl.RGB,self.FLOAT,null);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[1][i][1]);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE);
                this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_T,this.gl.CLAMP_TO_EDGE);
                this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[i+1][0],this.layerResolutions[i+1][1],0,this.gl.RGB,self.FLOAT,null);
                this.WaB.push(C);
                if (ops.CanTrain) {
					this.WaBdelta.push(D);
				}
            }
        }
        this.cycle = function(inpv) {
            this.gl.useProgram(this.cycleLayer);
            this.gl.bindTexture(this.gl.TEXTURE_2D,this.inputTextures[0]);
            this.gl.texImage2D(this.gl.TEXTURE_2D,0,this.gl.RGB,this.layerResolutions[0][0],this.layerResolutions[0][1],0,this.gl.RGB,this.gl.UNSIGNED_BYTE,inpv);
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER,this.cycleframeBuffer);
            for (var i=0;i<this.WaB.length;i++) {
                this.gl.scissor(0,0,this.layerResolutions[i+1][0],this.layerResolutions[i+1][1]);
                this.gl.viewport(0,0,this.layerResolutions[i+1][0],this.layerResolutions[i+1][1]);
                this.gl.activeTexture(this.gl.TEXTURE0);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][0]);
                this.gl.activeTexture(this.gl.TEXTURE1);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][1]);
                this.gl.activeTexture(this.gl.TEXTURE2);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][2]);
                this.gl.activeTexture(this.gl.TEXTURE3);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][3]);
                this.gl.activeTexture(this.gl.TEXTURE4);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][4]);
                this.gl.activeTexture(this.gl.TEXTURE5);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][5]);
                this.gl.uniform2iv(this.KernelSizeUniform,this.layerKernelSizes[i]);
                this.gl.activeTexture(this.gl.TEXTURE6);
                this.gl.bindTexture(this.gl.TEXTURE_2D,(i!==0) ? this.activations[0][i-1][0] : this.inputTextures[0]);
                this.gl.activeTexture(this.gl.TEXTURE7);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[1][i][0]);
                this.gl.drawElements(this.gl.TRIANGLES,6,this.gl.UNSIGNED_SHORT,0);
                this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[0][i][0]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i+1][0],this.layerResolutions[i+1][1],0);
                this.gl.readBuffer(this.gl.COLOR_ATTACHMENT1);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[0][i][1]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i+1][0],this.layerResolutions[i+1][1],0);
            }
			//this.activations.unshift(this.activations[2]);
			//this.activations.pop();
            this.activations.reverse();
            this.inputTextures.reverse();
        }
        this.setInput = function(inpv) {
            this.gl.bindTexture(this.gl.TEXTURE_2D,this.inputTextures[0]);
            this.gl.texImage2D(this.gl.TEXTURE_2D,0,this.gl.RGB,this.layerResolutions[0][0],this.layerResolutions[0][1],0,this.gl.RGB,this.gl.UNSIGNED_BYTE,inpv);
            this.inputTextures.reverse();
        }
        this.trainGradientWithCurrentActivations = function(grad) {
            this.DataSetSize++;
            for (var i=this.WaB.length-1;i>0;i--) {
				this.gl.bindFramebuffer(this.gl.FRAMEBUFFER,this.WaBframeBuffer);
                this.gl.useProgram(this.backpropWaB);
				this.gl.uniform1i(this.IsLastBackpropUniform,false);
                this.gl.uniform2iv(this.BackpropKernelSizeUniform,this.layerKernelSizes[i]);
                this.gl.scissor(0,0,this.layerResolutions[i][0],this.layerResolutions[i][1]);
                this.gl.viewport(0,0,this.layerResolutions[i][0],this.layerResolutions[i][1]);
                this.gl.activeTexture(this.gl.TEXTURE0);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][0]);
                this.gl.activeTexture(this.gl.TEXTURE1);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][1]);
                this.gl.activeTexture(this.gl.TEXTURE2);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][2]);
                this.gl.activeTexture(this.gl.TEXTURE3);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][3]);
                this.gl.activeTexture(this.gl.TEXTURE4);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][4]);
                this.gl.activeTexture(this.gl.TEXTURE5);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][5]);
                this.gl.activeTexture(this.gl.TEXTURE6);
                this.gl.bindTexture(this.gl.TEXTURE_2D,(i>=(this.WaB.length-1)) ? grad[0] : this.backpropTextures[i][0]);
                this.gl.activeTexture(this.gl.TEXTURE7);
                this.gl.bindTexture(this.gl.TEXTURE_2D,(i>=(this.WaB.length-1)) ? grad[1] : this.backpropTextures[i][1]);
                this.gl.activeTexture(this.gl.TEXTURE8);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[0][i][0]);
                this.gl.activeTexture(this.gl.TEXTURE9);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[0][i][1]);
                this.gl.activeTexture(this.gl.TEXTURE10);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[1][i-1][0]);
                this.gl.activeTexture(this.gl.TEXTURE11);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[1][i-1][1]);
                this.gl.drawElements(this.gl.TRIANGLES,6,this.gl.UNSIGNED_SHORT,0);
                this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][0]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i][0],this.layerResolutions[i][1],0);
                this.gl.readBuffer(this.gl.COLOR_ATTACHMENT1);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][1]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i][0],this.layerResolutions[i][1],0);
                this.gl.readBuffer(this.gl.COLOR_ATTACHMENT2);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][2]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i][0],this.layerResolutions[i][1],0);
                this.gl.readBuffer(this.gl.COLOR_ATTACHMENT3);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][3]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i][0],this.layerResolutions[i][1],0);
                this.gl.readBuffer(this.gl.COLOR_ATTACHMENT4);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][4]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i][0],this.layerResolutions[i][1],0);
                this.gl.readBuffer(this.gl.COLOR_ATTACHMENT5);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][5]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i][0],this.layerResolutions[i][1],0);
				
				
				this.gl.useProgram(this.backprop);
				this.gl.uniform2iv(this.backpropBackKernelSizeUniform,this.layerKernelSizes[i]);
				this.gl.bindFramebuffer(this.gl.FRAMEBUFFER,this.backpropFrameBuffer);
				this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0,this.gl.COLOR_ATTACHMENT1]);
				this.gl.bindTexture(this.gl.TEXTURE_2D,this.backpropTextures[i-1][0]);
				this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER,this.gl.COLOR_ATTACHMENT0,this.gl.TEXTURE_2D,this.backpropTextures[i-1][0],0);
				this.gl.bindTexture(this.gl.TEXTURE_2D,this.backpropTextures[i-1][1]);
				this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER,this.gl.COLOR_ATTACHMENT1,this.gl.TEXTURE_2D,this.backpropTextures[i-1][1],0);
				this.gl.activeTexture(this.gl.TEXTURE0);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][0]);
                this.gl.activeTexture(this.gl.TEXTURE1);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][1]);
                this.gl.activeTexture(this.gl.TEXTURE2);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][2]);
				this.gl.activeTexture(this.gl.TEXTURE3);
                this.gl.bindTexture(this.gl.TEXTURE_2D,(i>=(this.WaB.length-1)) ? grad[0] : this.backpropTextures[i][0]);
                this.gl.activeTexture(this.gl.TEXTURE4);
                this.gl.bindTexture(this.gl.TEXTURE_2D,(i>=(this.WaB.length-1)) ? grad[1] : this.backpropTextures[i][1]);
                this.gl.activeTexture(this.gl.TEXTURE5);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[1][i][0]);
				this.gl.drawElements(this.gl.TRIANGLES,6,this.gl.UNSIGNED_SHORT,0);
            }
			this.gl.bindFramebuffer(this.gl.FRAMEBUFFER,this.WaBframeBuffer);
			this.gl.useProgram(this.backpropWaB);
			this.gl.uniform1i(this.IsLastBackpropUniform,true);
			this.gl.uniform2iv(this.BackpropKernelSizeUniform,this.layerKernelSizes[0]);
			this.gl.scissor(0,0,this.layerResolutions[0][0],this.layerResolutions[0][1]);
			this.gl.viewport(0,0,this.layerResolutions[0][0],this.layerResolutions[0][1]);
			this.gl.activeTexture(this.gl.TEXTURE0);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[0][0]);
			this.gl.activeTexture(this.gl.TEXTURE1);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[0][1]);
			this.gl.activeTexture(this.gl.TEXTURE2);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[0][2]);
			this.gl.activeTexture(this.gl.TEXTURE3);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[0][3]);
			this.gl.activeTexture(this.gl.TEXTURE4);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[0][4]);
			this.gl.activeTexture(this.gl.TEXTURE5);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[0][5]);
			this.gl.activeTexture(this.gl.TEXTURE6);
			this.gl.bindTexture(this.gl.TEXTURE_2D,(0>=(this.WaB.length-1)) ? grad[0] : this.backpropTextures[0][0]);
			this.gl.activeTexture(this.gl.TEXTURE7);
			this.gl.bindTexture(this.gl.TEXTURE_2D,(0>=(this.WaB.length-1)) ? grad[1] : this.backpropTextures[0][1]);
			this.gl.activeTexture(this.gl.TEXTURE8);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[0][i][0]);
			this.gl.activeTexture(this.gl.TEXTURE9);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[0][i][1]);
			this.gl.activeTexture(this.gl.TEXTURE10);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.inputTextures[0]);
			this.gl.activeTexture(this.gl.TEXTURE11);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.inputTextures[1]);
			this.gl.drawElements(this.gl.TRIANGLES,6,this.gl.UNSIGNED_SHORT,0);
			this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[0][0]);
			this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[0][0],this.layerResolutions[0][1],0);
			this.gl.readBuffer(this.gl.COLOR_ATTACHMENT1);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[0][1]);
			this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[0][0],this.layerResolutions[0][1],0);
			this.gl.readBuffer(this.gl.COLOR_ATTACHMENT2);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[0][2]);
			this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[0][0],this.layerResolutions[0][1],0);
			this.gl.readBuffer(this.gl.COLOR_ATTACHMENT3);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[0][3]);
			this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[0][0],this.layerResolutions[0][1],0);
			this.gl.readBuffer(this.gl.COLOR_ATTACHMENT4);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[0][4]);
			this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[0][0],this.layerResolutions[0][1],0);
			this.gl.readBuffer(this.gl.COLOR_ATTACHMENT5);
			this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[0][5]);
			this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[0][0],this.layerResolutions[0][1],0);
        }
        this.resetZeros = function() {
            for (let i=0; i<this.layerKernelSizes.length; i++){
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[0][i]);
                this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[i+1][0],this.layerResolutions[i+1][1],0,this.gl.RGB,self.FLOAT,null);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[1][i]);
                this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[i+1][0],this.layerResolutions[i+1][1],0,this.gl.RGB,self.FLOAT,null);
            }
        }
        this.finishEpoch = function(lr) {
            this.gl.useProgram(this.addWaB);
            this.gl.uniform1f(this.addWabMult,lr/(this.DataSetSize));
            //this.gl.uniform1f(this.addWabMultW,lr/this.DataSetSize);
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER,this.WaBframeBuffer);
            for (var i=0;i<this.WaB.length;i++) {
                this.gl.scissor(0,0,this.layerResolutions[i][0],this.layerResolutions[i][1]);
                this.gl.viewport(0,0,this.layerResolutions[i][0],this.layerResolutions[i][1]);
                //this.gl.uniform1f(this.addWabMult,lr/(this.DataSetSize*Math.sqrt(this.layerKernelSizes[i][0]*this.layerKernelSizes[i][1]*6)));
				//this.gl.uniform1f(this.addWabMultW,lr/(this.DataSetSize*Math.sqrt(this.layerKernelSizes[i][0]*this.layerKernelSizes[i][1]*6)));
                this.gl.uniform2iv(this.addWabKernelSize,this.layerKernelSizes[i]);
                this.gl.activeTexture(this.gl.TEXTURE0);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][0]);
                this.gl.activeTexture(this.gl.TEXTURE1);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][1]);
                this.gl.activeTexture(this.gl.TEXTURE2);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][2]);
                this.gl.activeTexture(this.gl.TEXTURE3);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][3]);
                this.gl.activeTexture(this.gl.TEXTURE4);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][4]);
                this.gl.activeTexture(this.gl.TEXTURE5);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][5]);
                this.gl.activeTexture(this.gl.TEXTURE6);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][0]);
                this.gl.activeTexture(this.gl.TEXTURE7);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][1]);
                this.gl.activeTexture(this.gl.TEXTURE8);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][2]);
                this.gl.activeTexture(this.gl.TEXTURE9);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][3]);
                this.gl.activeTexture(this.gl.TEXTURE10);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][4]);
                this.gl.activeTexture(this.gl.TEXTURE11);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][5]);
                this.gl.drawElements(this.gl.TRIANGLES,6,this.gl.UNSIGNED_SHORT,0);
                this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][0]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i][0],this.layerResolutions[i][1],0);
                this.gl.readBuffer(this.gl.COLOR_ATTACHMENT1);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][1]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i][0],this.layerResolutions[i][1],0);
                this.gl.readBuffer(this.gl.COLOR_ATTACHMENT2);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][2]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i][0],this.layerResolutions[i][1],0);
                this.gl.readBuffer(this.gl.COLOR_ATTACHMENT3);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][3]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i][0],this.layerResolutions[i][1],0);
                this.gl.readBuffer(this.gl.COLOR_ATTACHMENT4);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][4]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i][0],this.layerResolutions[i][1],0);
                this.gl.readBuffer(this.gl.COLOR_ATTACHMENT5);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaB[i][5]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D,0,self.RGBF,0,0,this.layerResolutions[i][0],this.layerResolutions[i][1],0);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][0]);
                this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[i][0],this.layerResolutions[i][1],0,this.gl.RGB,self.FLOAT,null);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][1]);
                this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[i][0],this.layerResolutions[i][1],0,this.gl.RGB,self.FLOAT,null);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][2]);
                this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[i][0],this.layerResolutions[i][1],0,this.gl.RGB,self.FLOAT,null);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][3]);
                this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[i][0],this.layerResolutions[i][1],0,this.gl.RGB,self.FLOAT,null);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][4]);
                this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[i][0],this.layerResolutions[i][1],0,this.gl.RGB,self.FLOAT,null);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.WaBdelta[i][5]);
                this.gl.texImage2D(this.gl.TEXTURE_2D,0,self.RGBF,this.layerResolutions[i][0],this.layerResolutions[i][1],0,this.gl.RGB,self.FLOAT,null);
            }
            this.DataSetSize = 0;
        }
        this.textureToArray = function(tex,res) {
            this.gl.viewport(0,0,res[0],res[1]);
            this.gl.scissor(0,0,res[0],res[1]);
            this.gl.useProgram(drawPgrm);
            this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
            this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
            var data = new Float32Array(res[0]*res[1]*4);
            this.gl.readPixels(0,0,res[0],res[1],this.gl.RGBA,self.FLOAT,data);
            var result = new Float32Array(res[0]*res[1]*3);
            for (var i=0;i<res[0]*res[1];i++) {
                result.set(data.subarray(i*4,i*4+3),i*3);
            }
            return result;
        }
    }
    this.rcnn = new this.RCNN();
    this.rcnn.initalize(this.opts.Vision);
    this.RNN = function() {
        this.gl = self.gl;
        this.randomWaB = createProgram(`#version 300 es
            uniform highp mat4 uRand;
            uniform highp float uM;
            const highp float PI = 3.1415926535897932384626433832795;
            const highp float Tau = PI * 2.0;
            const highp float epsilon = 0.0000152587;
            highp vec4 seed = vec4(0.0,0.0,0.0,0.0);
            out highp vec4 fragColor;
            highp vec4 random() {
                highp uvec4 x = floatBitsToUint(seed);
                x = ((x>>8U)^x.ywzx)*110351524U;
                x = ((x>>8U)^x.ywzx)*110351524U;
                x = ((x>>8U)^x.ywzx)*110351524U;
                seed = mod((vec4(x)/314159.0)*uRand,1.0);
                //seed = mod((vec4(x)/3141593.0)*uRand,2.000000238418579)-1.0;
                return seed;
            }
            highp vec4 randn() {
                highp vec4 r0 = random();
                highp vec4 r1 = random();
                seed = clamp(sqrt(-2.0*log(r0))*cos(Tau*r1),vec4(-16.0),vec4(16.0));
                return seed;
            }
            void main(){
                seed = gl_FragCoord.xyzw*uRand;
                randn();
                random();
				fragColor = randn()*uM;
            }
        `);
        this.flatten = createProgram(`#version 300 es
            uniform highp sampler2D uInp[2];
            uniform highp sampler2D uExInp[2];
            uniform highp ivec2 uDims;
            out highp vec4 fragOut;
            void main(){
                highp int x = int(gl_FragCoord.x-0.5);
                highp ivec3 p = ivec3((x/3)%uDims[0],(x/3)/uDims[0],x%3);
                highp int l = 3*uDims[0]*uDims[1];
                if (x < l) {
                    fragOut = vec4(texelFetch(uInp[0],p.xy,0)[p.z],texelFetch(uInp[1],p.xy,0)[p.z],0,1);
					//fragOut = vec4(texelFetch(uInp[0],p.xy,0)[p.z],0,0,1);
                } else {
                    highp float extp = texelFetch(uExInp[0],ivec2(x-l,0),0).x;
                    fragOut = vec4(extp,extp-texelFetch(uExInp[1],ivec2(x-l,0),0).x,0,1);
					//fragOut = vec4(extp,0,0,1);
                }
            }
        `);
        this.cycleLayer = createProgram(`#version 300 es
            uniform highp sampler2D uWaB;
            uniform highp sampler2D uInpPrev;
            uniform highp sampler2D uInp;
            uniform highp int uLayerLength;
            uniform highp int uLayerInputTotal;
			uniform bool useActFunct;
            out highp vec4 Activation;
            highp float activationFunction(highp float v){
                // v = exp(-2.0*v);
                // return (1.0-v)/(1.0+v);

                // return tanh(v);

                return max(v,0.0);

                // return v/sqrt(1.0+(v*v));
            }
            void main(){
                highp float val = 0.0;
                highp int x = int(gl_FragCoord.x-0.5);
                highp vec4 w = texelFetch(uWaB,ivec2(x,0),0);
                val += w.x;
                for (highp int i = 0; i < uLayerInputTotal; i++){
                    highp int y = (i+1)/4;
                    highp int ym = (i+1)%4;
                    if (ym == 0) {
                        w = texelFetch(uWaB,ivec2(x,y),0);
                    }
                    if (i < uLayerLength) {
                        val += w[ym]*texelFetch(uInp,ivec2(i,0),0).x;
                    } else {
                        val += w[ym]*texelFetch(uInpPrev,ivec2(i-uLayerLength,0),0).x;
                    }
                }
				highp float prev = texelFetch(uInpPrev,ivec2(x,0),0).x;
				if (useActFunct) {
					val = activationFunction(val);
				}
				Activation = vec4(val,val-prev,0,1);
            }
        `);
        this.backprop = createProgram(`#version 300 es
            uniform highp sampler2D uWaB;
            uniform highp sampler2D uGrad;
            uniform highp sampler2D uActivations;
            uniform highp int uLayerLength;
            out highp vec4 Activation;
            highp float activationFunctionDerivitive(highp float v){
                // return 1.0-(v*v);
                return sign(v);
            }
            void main(){
                highp vec2 val;
                highp int x = int(gl_FragCoord.x-0.5);
                highp int y = (x+1)/4;
                highp int ym = (x+1)%4;
                for (highp int i = 0; i < uLayerLength; i++){
                    val += texelFetch(uGrad,ivec2(i,0),0).xy*texelFetch(uWaB,ivec2(i,y),0)[ym];
                }
                Activation = vec4(val*activationFunctionDerivitive(texelFetch(uActivations,ivec2(x,0),0).x),0,1);
				//Activation = vec4(val.x*activationFunctionDerivitive(texelFetch(uActivations,ivec2(x,0),0).x),val.y,0,1);
            }
        `);
        this.backpropWaB = createProgram(`#version 300 es
            uniform highp sampler2D uGrad;
            uniform highp sampler2D uWaBdelta;
            uniform highp sampler2D uInpPrev;
            uniform highp sampler2D uInp;
            uniform highp int uLayerLength;
            out highp vec4 newWaB;
            void main(){
                highp ivec2 xy = ivec2(gl_FragCoord.xy-0.5);
                highp vec2 vals = texelFetch(uGrad,ivec2(xy.x,0),0).xy;
                highp int y = (xy.y*4)-1;
                highp vec4 w;
                if (y == -1) {
                    w.x = vals.x;
                } else {
                    if (y < uLayerLength) {
						highp vec2 I = texelFetch(uInp,ivec2(y,0),0).xy;
                        w.x = dot(I,vals);
                    } else {
						highp vec2 I = texelFetch(uInpPrev,ivec2(y-uLayerLength,0),0).xy;
                        w.x = dot(I,vals);
                    }
                }
                y++;
                if (y < uLayerLength) {
					highp vec2 I = texelFetch(uInp,ivec2(y,0),0).xy;
                    w.y = dot(I,vals);
                } else {
					highp vec2 I = texelFetch(uInpPrev,ivec2(y-uLayerLength,0),0).xy;
                    w.y = dot(I,vals);
                }
                y++;
                if (y < uLayerLength) {
                    highp vec2 I = texelFetch(uInp,ivec2(y,0),0).xy;
                    w.z = dot(I,vals);
                } else {
                    highp vec2 I = texelFetch(uInpPrev,ivec2(y-uLayerLength,0),0).xy;
                    w.z = dot(I,vals);
                }
                y++;
                if (y < uLayerLength) {
                    highp vec2 I = texelFetch(uInp,ivec2(y,0),0).xy;
                    w.w = dot(I,vals);
                } else {
                    highp vec2 I = texelFetch(uInpPrev,ivec2(y-uLayerLength,0),0).xy;
                    w.w = dot(I,vals);
                }
                newWaB = w + texelFetch(uWaBdelta,xy,0);
            }
        `);
        this.addWaB = createProgram(`#version 300 es
            uniform highp sampler2D uWaB;
            uniform highp sampler2D uWaBdelta;
			uniform highp float uLerningRate;
            out highp vec4 newWaB;
            void main(){
                highp ivec2 xy = ivec2(gl_FragCoord.xy-0.5);
				newWaB = clamp(texelFetch(uWaB,xy,0) + texelFetch(uWaBdelta,xy,0)*uLerningRate,vec4(-16.0),vec4(16.0));
            }
        `);
        this.findDelta = createProgram(`#version 300 es
            uniform highp sampler2D uOutput;
            uniform highp sampler2D uTarget;
			uniform highp sampler2D uTarget2;
            out highp vec4 grad;
            highp float activationFunctionDerivitive(highp float v){
                // return 1.0-(v*v);
                return sign(v);
            }
            void main(){
                highp ivec2 xy = ivec2(gl_FragCoord.x-0.5,0);
                highp vec2 val = texelFetch(uOutput,xy,0).xy;
                highp float target = texelFetch(uTarget,xy,0).x;
				highp float target2 = texelFetch(uTarget2,xy,0).x;
				//grad = vec4(target-val.x,(target-target2),0,1);
				grad = vec4(target-val.x,((target-target2)-val.y),0,1);
				
				//val.y += target2;
				//grad = vec4(vec2(target)-val,0,1);
				
                //grad = vec4(0,((target-target2)-val.y),0,1);
                //grad = vec4((val-target)*activationFunctionDerivitive(val));
            }
        `);
		this.findReifDelta = createProgram(`#version 300 es
            uniform highp sampler2D uOutput;
            uniform highp float uMult;
            out highp vec4 grad;
            highp float activationFunctionDerivitive(highp float v){
                // return 1.0-(v*v);
                return sign(v);
            }
            void main(){
                highp ivec2 xy = ivec2(gl_FragCoord.x-0.5,0);
                highp vec2 val = texelFetch(uOutput,xy,0).xy;
				// grad = vec4(val*uMult);
                grad = vec4(sign(val.x)*uMult,val.y*uMult,0,1);
                // grad = vec4(val*uMult*activationFunctionDerivitive(val));
                //grad = vec4(sign(val)*uMult*activationFunctionDerivitive(val));
            }
        `);
        this.backpropUnflatten = createProgram(`#version 300 es
            uniform highp sampler2D uWaB;
            uniform highp sampler2D uGrads;
            uniform highp sampler2D uInptVals;
            uniform highp int uLayerLength;
            uniform highp ivec2 uDims;
            out highp vec4 Activation[2];
            highp vec3 activationFunctionDerivitive(highp vec3 v){
                // return 1.0-(v*v);
                return sign(v);
            }
            void main(){
                highp vec3 val = vec3(0.0);
                highp vec3 val2 = vec3(0.0);
                highp ivec2 xy = ivec2(gl_FragCoord.xy-0.5);
                highp int x = (xy.x+(xy.y*uDims[0]))*3;
                highp int y0 = (x+1)/4;
                highp int ym0 = (x+1)%4;
				highp int y1 = (x+2)/4;
                highp int ym1 = (x+2)%4;
				highp int y2 = (x+3)/4;
                highp int ym2 = (x+3)%4;
                for (highp int i = 0; i < uLayerLength; i++){
                    highp vec2 g = texelFetch(uGrads,ivec2(i,0),0).xy;
                    highp vec3 w = vec3(texelFetch(uWaB,ivec2(i,y0),0)[ym0],texelFetch(uWaB,ivec2(i,y1),0)[ym1],texelFetch(uWaB,ivec2(i,y2),0)[ym2]);
					val += g.x*w;
                    val2 += g.y*w;
                }
                highp vec3 driv = activationFunctionDerivitive(texelFetch(uInptVals,xy,0).xyz);
                Activation[0] = vec4(val*driv,1.0);
                Activation[1] = vec4(val2*driv,1.0);
            }
        `);
		this.calcAddLoss = createProgram(`#version 300 es
            uniform highp sampler2D uOutput;
            uniform highp sampler2D uTarget;
			uniform highp sampler2D uPrev;
            out highp vec4 grad;
            void main(){
                highp ivec2 xy = ivec2(int(gl_FragCoord.x-0.5),0);
                highp float val = texelFetch(uOutput,xy,0).x;
				highp float goal = texelFetch(uTarget,xy,0).x;
				//highp float val = (sign(texelFetch(uTarget,xy,0).x)-sign(texelFetch(uOutput,xy,0).x));
                //grad = vec4((val*val)+texelFetch(uPrev,xy,0).x);
				grad = vec4(abs(val-goal)+texelFetch(uPrev,xy,0).x);
				
				//highp float maximum = max(abs(val),abs(goal));
				//grad = vec4(abs((val-goal)/maximum)+texelFetch(uPrev,xy,0).x);
            }
        `);
        this.initalize = function(ops) {
            this.inpShape = ops.inpShape;
            this.inpSize = ops.inpSize;
            this.WaB = [];
            this.WaBdelta = [];
            this.activations = [[],[]];
            this.backpropTextures = [];
            this.extraInp = ops.ExtraInputs;
            this.fullInpSize = this.inpSize + this.extraInp;
            this.DataSetSize = 0;
            this.NumberOfParamiters = 0;
            this.outputSize = ops.layers[ops.layers.length-1];
			this.layers = [this.fullInpSize].concat(ops.layers);
			this.lossTexture = this.gl.createTexture();
			this.gl.bindTexture(this.gl.TEXTURE_2D, this.lossTexture);
			this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
			this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
			this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
			this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
			this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RF, this.outputSize, 1, 0, this.gl.RED, self.FLOAT, null);
            this.unflattenframeBuffer = this.gl.createFramebuffer();
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.unflattenframeBuffer);
            this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0,this.gl.COLOR_ATTACHMENT1]);
            this.unflattenTexture0 = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.unflattenTexture0);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RGBAF, this.inpShape[0], this.inpShape[1], 0, this.gl.RGBA, self.FLOAT, null);
            this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, this.unflattenTexture0, 0);
            this.unflattenTexture1 = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.unflattenTexture1);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RGBAF, this.inpShape[0], this.inpShape[1], 0, this.gl.RGBA, self.FLOAT, null);
            this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT1, this.gl.TEXTURE_2D, this.unflattenTexture1, 0);
            this.flattenTexture = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.flattenTexture);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RGF, this.fullInpSize, 1, 0, this.gl.RG, self.FLOAT, null);
            this.targetTexture = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.targetTexture);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RF, this.outputSize, 1, 0, this.gl.RED, self.FLOAT, null);
			this.targetTexture2 = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.targetTexture2);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RF, this.outputSize, 1, 0, this.gl.RED, self.FLOAT, null);
			this.targetTextures = [this.targetTexture,this.targetTexture2];
            this.extraInputTexture0 = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.extraInputTexture0);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RF, this.extraInp, 1, 0, this.gl.RED, self.FLOAT, null);
            this.extraInputTexture1 = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.extraInputTexture1);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RF, this.extraInp, 1, 0, this.gl.RED, self.FLOAT, null);
            this.extraInputTextures = [this.extraInputTexture0,this.extraInputTexture1];
            var lyrz = this.layers;
            var maxWid = this.layers.reduce(function(a,b) {return Math.max(a,b)});
            this.WaBorigHeights = this.layers.slice(0,-1).map(function(x,xi){return x+lyrz[xi]+1;});
            this.WaBorigResolutions = this.WaBorigHeights.map(function(x,xi){return [lyrz[xi+1],x];});
            this.WaBheights = this.WaBorigHeights.map(function(x){return Math.ceil(x/4)});
            this.WaBresolutions = this.WaBorigResolutions.map(function(x){return [x[0],Math.ceil(x[1]/4)];});
            var maxHet = this.WaBheights.reduce(function(a,b){return Math.max(a,b)});
            // this.layers = ops.layers;
            this.frameBuffer = this.gl.createFramebuffer();
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
            this.frameBufferTexture = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.frameBufferTexture);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RGBAF, maxWid, maxHet, 0, this.gl.RGBA, self.FLOAT, null);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
            this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, this.frameBufferTexture, 0);
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
            this.gl.useProgram(this.cycleLayer);
            this.gl.uniform1i(this.gl.getUniformLocation(this.cycleLayer, "uWaB"), 0);
            this.gl.uniform1i(this.gl.getUniformLocation(this.cycleLayer, "uInp"), 1);
            this.gl.uniform1i(this.gl.getUniformLocation(this.cycleLayer, "uInpPrev"), 2);
            this.useActFunct = this.gl.getUniformLocation(this.cycleLayer, "useActFunct");
			this.LayerLengthUniform = this.gl.getUniformLocation(this.cycleLayer, "uLayerLength");
            this.LayerInputTotalUniform = this.gl.getUniformLocation(this.cycleLayer, "uLayerInputTotal");
            this.gl.useProgram(this.flatten);
            this.gl.uniform1iv(this.gl.getUniformLocation(this.flatten, "uInp"), [0,1]);
            this.gl.uniform1iv(this.gl.getUniformLocation(this.flatten, "uExInp"), [2,3]);
            this.FlattenDimsUniform = this.gl.getUniformLocation(this.flatten, "uDims");
            this.gl.useProgram(this.backprop);
            this.gl.uniform1i(this.gl.getUniformLocation(this.backprop, "uWaB"), 0);
            this.gl.uniform1i(this.gl.getUniformLocation(this.backprop, "uGrad"), 1);
            this.gl.uniform1i(this.gl.getUniformLocation(this.backprop, "uActivations"), 2);
            this.BackpropLayerLengthUniform = this.gl.getUniformLocation(this.backprop, "uLayerLength");
            this.gl.useProgram(this.backpropWaB);
            this.gl.uniform1i(this.gl.getUniformLocation(this.backpropWaB, "uWaBdelta"), 0);
            this.gl.uniform1i(this.gl.getUniformLocation(this.backpropWaB, "uGrad"), 1);
            this.gl.uniform1i(this.gl.getUniformLocation(this.backpropWaB, "uInp"), 2);
            this.gl.uniform1i(this.gl.getUniformLocation(this.backpropWaB, "uInpPrev"), 3);
            this.BackpropWaBLayerLengthUniform = this.gl.getUniformLocation(this.backpropWaB, "uLayerLength");
			//uRegM
            this.gl.useProgram(this.addWaB);
            this.gl.uniform1i(this.gl.getUniformLocation(this.addWaB, "uWaB"), 0);
            this.gl.uniform1i(this.gl.getUniformLocation(this.addWaB, "uWaBdelta"), 1);
			this.LerningRateUniform = this.gl.getUniformLocation(this.addWaB, "uLerningRate");
            this.gl.useProgram(this.findDelta);
            this.gl.uniform1i(this.gl.getUniformLocation(this.findDelta, "uTarget"), 0);
			this.gl.uniform1i(this.gl.getUniformLocation(this.findDelta, "uTarget2"), 1);
            this.gl.uniform1i(this.gl.getUniformLocation(this.findDelta, "uOutput"), 2);
			//findReifDelta
			this.gl.useProgram(this.findReifDelta);
            this.gl.uniform1i(this.gl.getUniformLocation(this.findReifDelta, "uOutput"), 0);
            this.ReinforcementUiniform = this.gl.getUniformLocation(this.findReifDelta, "uMult");
			this.gl.useProgram(this.calcAddLoss);
            this.gl.uniform1i(this.gl.getUniformLocation(this.calcAddLoss, "uTarget"), 0);
            this.gl.uniform1i(this.gl.getUniformLocation(this.calcAddLoss, "uOutput"), 1);
            this.gl.uniform1i(this.gl.getUniformLocation(this.calcAddLoss, "uPrev"), 2);
            this.gl.useProgram(this.backpropUnflatten);
            this.gl.uniform1i(this.gl.getUniformLocation(this.backpropUnflatten, "uWaB"), 0);
            this.gl.uniform1i(this.gl.getUniformLocation(this.backpropUnflatten, "uGrads"), 1);
            this.gl.uniform1i(this.gl.getUniformLocation(this.backpropUnflatten, "uInptVals"), 2);
            this.BackpropUnflattenDimsUniform = this.gl.getUniformLocation(this.backpropUnflatten, "uDims");
            this.BackpropUnflattenLayerLengthUniform = this.gl.getUniformLocation(this.backpropUnflatten, "uLayerLength");
            this.gl.useProgram(this.randomWaB);
            var m = this.gl.getUniformLocation(this.randomWaB,"uM");
            var uRand = this.gl.getUniformLocation(this.randomWaB,"uRand");
            for (var i=0;i<this.WaBresolutions.length;i++) {
                var res = this.WaBresolutions[i];
                this.NumberOfParamiters += this.WaBorigResolutions[i][0]*this.WaBorigResolutions[i][1];
                //this.gl.uniform1f(m,2/Math.sqrt(this.WaBorigHeights[i]));
				this.gl.uniform1f(m,Math.sqrt(1/this.WaBorigHeights[i]));
				//this.gl.uniform1f(m,0.625/Math.sqrt(this.layers[i]));
				//this.gl.uniform1f(m,Math.sqrt(1/this.layers[i]));
				//this.gl.uniform1i(this.RandomWaBLengthUniform,this.layers[i]+1);
				//this.RandomWaBLengthUniform
                //this.gl.uniform1f(m,1);
                this.gl.uniformMatrix4fv(uRand,false,new Float32Array(16).map(function(){return (Math.random()-0.5)*20}));
                this.gl.scissor(0,0,res[0],res[1]);
                this.gl.viewport(0,0,res[0],res[1]);
                this.WaB.push(this.gl.createTexture());
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaB[i]);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
                this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RGBAF, res[0], res[1], 0, this.gl.RGBA, self.FLOAT, null);
                this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaB[i]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D, 0, self.RGBAF, 0, 0, res[0], res[1], 0);
				if (ops.CanTrain) {
					this.WaBdelta.push(this.gl.createTexture());
					this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaBdelta[i]);
					this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
					this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
					this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
					this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
					this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RGBAF, res[0], res[1], 0, this.gl.RGBA, self.FLOAT, null);
				}
                this.activations[0].push(this.gl.createTexture());
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.activations[0][i]);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
                this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RGF, this.layers[i+1], 1, 0, this.gl.RG, self.FLOAT, null);
                this.activations[1].push(this.gl.createTexture());
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.activations[1][i]);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
                this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
                this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RGF, this.layers[i+1], 1, 0, this.gl.RG, self.FLOAT, null);
				if (ops.CanTrain) {
					this.backpropTextures.push(this.gl.createTexture());
					this.gl.bindTexture(this.gl.TEXTURE_2D, this.backpropTextures[i]);
					this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
					this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
					this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
					this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
					this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RGF, this.layers[i+1], 1, 0, this.gl.RG, self.FLOAT, null);
				}
            }
        }
        this.cycle = function(inpw,exinp) {
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.extraInputTextures[0]);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.R32F, this.extraInp, 1, 0, this.gl.RED, this.gl.FLOAT, exinp);
            this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
            this.gl.useProgram(this.flatten);
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
            this.gl.scissor(0,0,this.fullInpSize,1);
            this.gl.viewport(0,0,this.fullInpSize,1);
            this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, inpw[0]);
            this.gl.activeTexture(this.gl.TEXTURE1);
            this.gl.bindTexture(this.gl.TEXTURE_2D, inpw[1]);
            this.gl.activeTexture(this.gl.TEXTURE2);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.extraInputTextures[0]);
            this.gl.activeTexture(this.gl.TEXTURE3);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.extraInputTextures[1]);
            this.gl.uniform2iv(this.FlattenDimsUniform, this.inpShape);
            this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.flattenTexture);
            this.gl.copyTexImage2D(this.gl.TEXTURE_2D, 0, self.RGF, 0, 0, this.fullInpSize, 1, 0);
            this.cycleFlattened(this.flattenTexture);
            this.extraInputTextures.reverse();
        }
        this.cycleFlattened = function(inpv) {
            this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
            this.gl.useProgram(this.cycleLayer);
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
            for (var i = 0; i < this.activations[0].length; i++) {
                this.gl.scissor(0,0,this.layers[i+1],1);
                this.gl.viewport(0,0,this.layers[i+1],1);
                this.gl.uniform1i(this.LayerLengthUniform, this.layers[i]);
                this.gl.uniform1i(this.LayerInputTotalUniform, this.WaBorigHeights[i]);
				this.gl.uniform1i(this.useActFunct, i!==(this.activations[0].length-1));
				//this.useActFunct
                this.gl.activeTexture(this.gl.TEXTURE0);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaB[i]);
                this.gl.activeTexture(this.gl.TEXTURE1);
                this.gl.bindTexture(this.gl.TEXTURE_2D, (this.activations[0][i-1] || inpv));
                this.gl.activeTexture(this.gl.TEXTURE2);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.activations[1][i]);
                this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.activations[0][i]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D, 0, self.RGF, 0, 0, this.layers[i+1], 1, 0);
            }
			//this.activations.unshift(this.activations[1]);
			//this.activations.pop();
            this.activations.reverse();
        }
        this.trainGradientWithCurrentGradientAndActivations = function() {
            this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
			this.gl.useProgram(this.backpropWaB);
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
            for (var i = this.WaB.length-1; i > 0; i--) {
                this.gl.useProgram(this.backpropWaB);
                var resol = this.WaBresolutions[i];
                this.gl.scissor(0,0,resol[0],resol[1]);
                this.gl.viewport(0,0,resol[0],resol[1]);
                this.gl.activeTexture(this.gl.TEXTURE0);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaBdelta[i]);
                this.gl.activeTexture(this.gl.TEXTURE1);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.backpropTextures[i]);
                this.gl.activeTexture(this.gl.TEXTURE2);
                this.gl.bindTexture(this.gl.TEXTURE_2D, (this.activations[1][i-1] || this.flattenTexture));
                this.gl.activeTexture(this.gl.TEXTURE3);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.activations[0][i]);
                this.gl.uniform1i(this.BackpropWaBLayerLengthUniform, this.layers[i]);
                this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaBdelta[i]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D, 0, self.RGBAF, 0, 0, resol[0], resol[1], 0);
                this.gl.useProgram(this.backprop);
                this.gl.scissor(0,0,this.layers[i],1);
                this.gl.viewport(0,0,this.layers[i],1);
                this.gl.activeTexture(this.gl.TEXTURE0);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaB[i]);
                this.gl.activeTexture(this.gl.TEXTURE1);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.backpropTextures[i]);
                this.gl.activeTexture(this.gl.TEXTURE2);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.activations[1][i-1]);
                this.gl.uniform1i(this.BackpropLayerLengthUniform, this.layers[i]);
                this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.backpropTextures[i-1]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D, 0, self.RGF, 0, 0, this.layers[i], 1, 0);
				//console.log(this.textureToArray2D(this.backpropTextures[i-1],[this.layers[i],1]));
            }
            this.gl.useProgram(this.backpropWaB);
            var resol = this.WaBresolutions[0];
            this.gl.scissor(0,0,resol[0],resol[1]);
            this.gl.viewport(0,0,resol[0],resol[1]);
            this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaBdelta[0]);
            this.gl.activeTexture(this.gl.TEXTURE1);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.backpropTextures[0]);
            this.gl.activeTexture(this.gl.TEXTURE2);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.flattenTexture);
            this.gl.activeTexture(this.gl.TEXTURE3);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.activations[1][0]);
            this.gl.uniform1i(this.BackpropWaBLayerLengthUniform, this.layers[0]);
            this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaBdelta[0]);
            this.gl.copyTexImage2D(this.gl.TEXTURE_2D, 0, self.RGBAF, 0, 0, resol[0], resol[1], 0);
            this.DataSetSize++;
        }
        this.train = function(inpw, exinp, outw) {
            this.cycle(inpw,exinp);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.targetTextures[0]);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.R32F, this.outputSize, 1, 0, this.gl.RED, this.gl.FLOAT, outw);
			this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
			this.gl.scissor(0,0,this.outputSize,1);
            this.gl.viewport(0,0,this.outputSize,1);
			this.gl.useProgram(this.calcAddLoss);
			this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.targetTexture);
            this.gl.activeTexture(this.gl.TEXTURE1);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.activations[1][this.activations[1].length-1]);
			this.gl.activeTexture(this.gl.TEXTURE2);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.lossTexture);
			this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
			this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
			this.gl.bindTexture(this.gl.TEXTURE_2D, this.lossTexture);
			this.gl.copyTexImage2D(this.gl.TEXTURE_2D, 0, self.RF, 0, 0, this.outputSize, 1, 0);
			//this.lossTexture
            this.gl.useProgram(this.findDelta);
			//this.gl.uniform1i(this.gl.getUniformLocation(this.findDelta, "uTarget"), 0);
            //this.gl.uniform1i(this.gl.getUniformLocation(this.findDelta, "uOutput"), 1);
			this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.targetTextures[0]);
			this.gl.activeTexture(this.gl.TEXTURE1);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.targetTextures[1]);
            this.gl.activeTexture(this.gl.TEXTURE2);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.activations[1][this.activations[1].length-1]);
            this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
			this.targetTextures.reverse();
            this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.backpropTextures[this.backpropTextures.length-1]);
            this.gl.copyTexImage2D(this.gl.TEXTURE_2D, 0, self.RGF, 0, 0, this.outputSize, 1, 0);
            this.trainGradientWithCurrentGradientAndActivations();
        }
        this.trainCurrentWithGradientOut = function(inpw, outw) {
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.targetTextures[0]);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.R32F, this.outputSize, 1, 0, this.gl.RED, this.gl.FLOAT, outw);
			this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
			this.gl.scissor(0,0,this.outputSize,1);
            this.gl.viewport(0,0,this.outputSize,1);
			this.gl.useProgram(this.calcAddLoss);
			this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.targetTextures[0]);
            this.gl.activeTexture(this.gl.TEXTURE1);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.activations[1][this.WaB.length-1]);
			this.gl.activeTexture(this.gl.TEXTURE2);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.lossTexture);
			this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
			this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
			this.gl.bindTexture(this.gl.TEXTURE_2D, this.lossTexture);
			this.gl.copyTexImage2D(this.gl.TEXTURE_2D, 0, self.RF, 0, 0, this.outputSize, 1, 0);
			
            this.gl.useProgram(this.findDelta);
			this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.targetTextures[0]);
			this.gl.activeTexture(this.gl.TEXTURE1);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.targetTextures[1]);
            this.gl.activeTexture(this.gl.TEXTURE2);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.activations[1][this.WaB.length-1]);
            this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.backpropTextures[this.backpropTextures.length-1]);
            this.gl.copyTexImage2D(this.gl.TEXTURE_2D, 0, self.RGF, 0, 0, this.outputSize, 1, 0);
			//console.log(this.textureToArray2D(this.backpropTextures[this.backpropTextures.length-1],[this.outputSize,1]));
            this.trainGradientWithCurrentGradientAndActivations();
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.unflattenframeBuffer);
            this.gl.useProgram(this.backpropUnflatten);
            var resol = this.inpShape;
            this.gl.scissor(0,0,resol[0],resol[1]);
            this.gl.viewport(0,0,resol[0],resol[1]);
            this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaB[0]);
            this.gl.activeTexture(this.gl.TEXTURE1);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.backpropTextures[0]);
            this.gl.activeTexture(this.gl.TEXTURE2);
            this.gl.bindTexture(this.gl.TEXTURE_2D, inpw);
            this.gl.uniform1i(this.BackpropUnflattenLayerLengthUniform, this.layers[1]);
            this.gl.uniform2iv(this.BackpropUnflattenDimsUniform, resol);
            this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
			this.targetTextures.reverse();
            return [this.unflattenTexture0,this.unflattenTexture1];
        }
		this.ReinforceTrainWithGradientOut = function(inpw, PositiveEnforcement) {
            // this.cycle(inpw,exinp);
			this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
			this.gl.scissor(0,0,this.outputSize,1);
            this.gl.viewport(0,0,this.outputSize,1);
			
            this.gl.useProgram(this.findReifDelta);
			this.gl.uniform1f(this.ReinforcementUiniform, PositiveEnforcement);
			this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.activations[1][this.activations[1].length-1]);
            this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
            this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.backpropTextures[this.backpropTextures.length-1]);
            this.gl.copyTexImage2D(this.gl.TEXTURE_2D, 0, self.RGF, 0, 0, this.outputSize, 1, 0);
            this.trainGradientWithCurrentGradientAndActivations();
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.unflattenframeBuffer);
            this.gl.useProgram(this.backpropUnflatten);
            var resol = this.inpShape;
            this.gl.scissor(0,0,resol[0],resol[1]);
            this.gl.viewport(0,0,resol[0],resol[1]);
            this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaB[0]);
            this.gl.activeTexture(this.gl.TEXTURE1);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.backpropTextures[0]);
            this.gl.activeTexture(this.gl.TEXTURE2);
            this.gl.bindTexture(this.gl.TEXTURE_2D, inpw);
            this.gl.uniform1i(this.BackpropUnflattenLayerLengthUniform, this.layers[1]);
            this.gl.uniform2iv(this.BackpropUnflattenDimsUniform, resol);
            this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
            return [this.unflattenTexture0,this.unflattenTexture1];
        }
        this.resetZeros = function() {
            for (let i=0; i<this.WaBresolutions.length; i++){
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[0][i]);
                this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RGF, this.layers[i+1], 1, 0, this.gl.RG, self.FLOAT, null);
                this.gl.bindTexture(this.gl.TEXTURE_2D,this.activations[1][i]);
                this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RGF, this.layers[i+1], 1, 0, this.gl.RG, self.FLOAT, null);
            }
        }
        this.finishEpoch = function(lr,returnloss) {
            this.gl.useProgram(this.addWaB);
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
			this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
            var ret = undefined;
			if (returnloss) {
				var lossArr = this.textureToArray1D(this.lossTexture,[this.outputSize,1]);
				ret = (lossArr.reduce(function(a,b) {return a+b;})/(this.outputSize*this.DataSetSize));
			}
            this.gl.useProgram(this.addWaB);
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
			this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
			this.gl.uniform1f(this.LerningRateUniform, lr/(this.DataSetSize));
            //this.gl.uniform1f(this.LerningRateUniform, lr/(this.outputSize*this.DataSetSize*2));
			//this.gl.uniform1f(this.LerningRateUniform, lr/(Math.sqrt(this.outputSize)*this.DataSetSize));
			// this.gl.uniform1f(this.LerningRateUniform, lr/(this.outputSize*Math.sqrt(this.DataSetSize)));
            for (var i = 0; i < this.WaB.length; i++) {
                var resol = this.WaBresolutions[i];
                this.gl.scissor(0,0,resol[0],resol[1]);
                this.gl.viewport(0,0,resol[0],resol[1]);
				//this.gl.uniform1f(this.LerningRateUniform, lr/(this.DataSetSize*this.outputSize*Math.sqrt(this.layers[i+1]/this.layers[i])));
                this.gl.activeTexture(this.gl.TEXTURE0);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaB[i]);
                this.gl.activeTexture(this.gl.TEXTURE1);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaBdelta[i]);
                this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaB[i]);
                this.gl.copyTexImage2D(this.gl.TEXTURE_2D, 0, self.RGBAF, 0, 0, resol[0], resol[1], 0);
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.WaBdelta[i]);
                this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RGBAF, resol[0], resol[1], 0, this.gl.RGBA, self.FLOAT, null);
				//this.gl.uniform1f(this.LerningRateUniform, lr/(this.DataSetSize*Math.sqrt(this.WaBorigHeights[i])));
            }
			this.gl.bindTexture(this.gl.TEXTURE_2D, this.lossTexture);
			this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RF, this.outputSize, 1, 0, this.gl.RED, self.FLOAT, null);
            this.DataSetSize = 0;
			return ret;
        }
        this.textureToArray1D = function(tex,res) {
            this.gl.viewport(0,0,res[0],res[1]);
            this.gl.scissor(0,0,res[0],res[1]);
            this.gl.useProgram(drawPgrm);
            this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
            this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
            if (self.Float16) {
                var data = new Uint16Array(res[0]*res[1]*4);
                this.gl.readPixels(0,0,res[0],res[1],this.gl.RGBA,self.FLOAT,data);
                var result = new Uint16Array(res[0]*res[1]);
                for (var i=0;i<res[0]*res[1];i++) {
                    result[i] = data[i*4];
                }
                return self.Uint16ArrayToFloat32Array(result);
            } else {
                var data = new Float32Array(res[0]*res[1]*4);
                this.gl.readPixels(0,0,res[0],res[1],this.gl.RGBA,self.FLOAT,data);
                var result = new Float32Array(res[0]*res[1]);
                for (var i=0;i<res[0]*res[1];i++) {
                    result[i] = data[i*4];
                }
                return result;
            }
        }
		this.textureToArray2D = function(tex,res) {
            this.gl.viewport(0,0,res[0],res[1]);
            this.gl.scissor(0,0,res[0],res[1]);
            this.gl.useProgram(drawPgrm);
            this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
            this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
            if (self.Float16) {
                var data = new Uint16Array(res[0]*res[1]*4);
                this.gl.readPixels(0,0,res[0],res[1],this.gl.RGBA,self.FLOAT,data);
                var result = new Uint16Array(res[0]*res[1]*2);
                for (var i=0;i<result.length;i+=2) {
                    result[i] = data[i*4];
					result[i+1] = data[(i*4)+1];
                }
                return self.Uint16ArrayToFloat32Array(result);
            } else {
                var data = new Float32Array(res[0]*res[1]*4);
                this.gl.readPixels(0,0,res[0],res[1],this.gl.RGBA,self.FLOAT,data);
                var result = new Float32Array(res[0]*res[1]*2);
                for (var i=0;i<result.length;i+=2) {
                    result[i] = data[i*2];
					result[i+1] = data[(i*2)+1];
                }
                return result;
            }
        }
        this.textureToArray4D = function(tex,res) {
            this.gl.viewport(0,0,res[0],res[1]);
            this.gl.scissor(0,0,res[0],res[1]);
            this.gl.useProgram(drawPgrm);
            this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
            this.gl.drawElements(this.gl.TRIANGLES, 6, this.gl.UNSIGNED_SHORT, 0);
            // var result = new Float32Array(res[0]*res[1]*4);
            if (self.Float16) {
                var result = new Uint16Array(res[0]*res[1]*4);
                this.gl.readPixels(0,0,res[0],res[1],this.gl.RGBA,self.FLOAT,result);
                return self.Uint16ArrayToFloat32Array(result);
            } else {
                var result = new Float32Array(res[0]*res[1]*4);
                this.gl.readPixels(0,0,res[0],res[1],this.gl.RGBA,self.FLOAT,result);
                return result;
            }
        }
    }
    this.rnn = new this.RNN();
    this.opts.Brain.inpSize = this.rcnn.layerResolutions[this.rcnn.layerResolutions.length-1][0]*this.rcnn.layerResolutions[this.rcnn.layerResolutions.length-1][1]*3;
    this.opts.Brain.inpShape = this.rcnn.layerResolutions[this.rcnn.layerResolutions.length-1];
    this.rnn.initalize(this.opts.Brain);
    this.NumberOfParamiters = this.rnn.NumberOfParamiters+this.rcnn.NumberOfParamiters;
    this.cycle = function(inpz,exinp,returnResult) {
        this.rcnn.cycle(inpz);
        this.rnn.cycle(this.rcnn.activations[1][this.rcnn.activations[1].length-1],exinp);
        if (returnResult) {
            return this.rnn.textureToArray1D(this.rnn.activations[1][this.rnn.activations[1].length-1],[this.rnn.layers[this.rnn.layers.length-1],1]);
        }
    }
    this.train = function(inpw, exinp, outw, returnloss) {
        self.cycle(inpw,exinp);
        return self.trainCurrent(outw, returnloss);
    }
    //trainCurrentWithGradientOut
    this.trainCurrent = function(outw, returnloss) {
        var grad = this.rnn.trainCurrentWithGradientOut(this.rcnn.activations[1][this.rcnn.activations[1].length-1][0],outw);
        this.rcnn.trainGradientWithCurrentActivations(grad);
        if (returnloss) {
            var AIout = this.rnn.textureToArray1D(this.rnn.activations[1][this.rnn.layers.length-2],[this.rnn.layers[this.rnn.layers.length-1],1]);
            // var loss = AIout.map(function(x,i) {x -= outw[i]; return x*x;}).reduce(function(a,b) {return a+b;})/(this.rnn.outputSize*2);
             var loss = AIout.map(function(x,i) {x -= outw[i]; return Math.abs(x);}).reduce(function(a,b) {return a+b;})/(this.rnn.outputSize);
            //var loss = Math.sqrt(AIout.map(function(x,i) {x -= outw[i]; return x*x;}).reduce(function(a,b) {return a+b;})/(this.rnn.outputSize));
            // return {Output:AIout,Loss:loss};
            return {Output:AIout,loss:loss};
        }
    }
	this.ReinforcementTrain = function(PositiveEnforcement) {
        var grad = this.rnn.ReinforceTrainWithGradientOut(this.rcnn.activations[1][this.rcnn.activations[1].length-1][0],PositiveEnforcement);
        this.rcnn.trainGradientWithCurrentActivations(grad);
    }
    this.resetZeros = function() {
        this.rnn.resetZeros();
        this.rcnn.resetZeros();
    }
    this.finishEpoch = function(lr,returnloss) {
        var ret = this.rnn.finishEpoch(lr,returnloss);
        this.rcnn.finishEpoch(lr);
		return ret;
    }
    this.Uint16ArrayToFloat32Array = function(data) {
        var result = new Float32Array(data.length);
        for (var i=0;i<data.length;i++) {
            var F16 = (data[i]%1024)/512;
            F16 *= 2**((Math.floor(data[i]/1024)%32)-15);
            F16 *= (Math.floor(data[i]/32768)==1)?-1:1;
            result[i] = F16;
        }
        return result;
    }
    this.Float32ArrayToUint16Array = function(data) {
        var result = new Uint16Array(data.length);
        for (var i=0;i<data.length;i++) {
            var U16 = 0;
            var exp = Math.ceil(Math.max(Math.log2(Math.abs(data[i]))+15,0));
            U16 = Math.floor(512*data[i]/(2**(exp-15)));
            U16 += exp*1024;
            U16 += (data[i]<0)?32768:0;
            result[i] = U16;
        }
        return result;
    }
    this.Export = function() {
        var result = new Float32Array(this.NumberOfParamiters);
        var idx = 0;
        var idxp = this.rcnn.layerResolutions[0][0]*this.rcnn.layerResolutions[0][1]*3;
        for (var i=0;i<this.rcnn.WaB.length;i++) {
            for (var j=0;j<this.rcnn.WaB[i];j++) {
                result.set(this.rcnn.textureToArray(this.rcnn.WaB[i][j],this.rcnn.layerResolutions[i]),idx);
                idx += idxp;
            }
        }
        for (var i=0;i<this.rnn.WaB.length;i++) {
            result.set(this.rnn.textureToArray4D(this.rnn.WaB[i],this.rnn.WaBresolutions[i]),idx);
            idx += this.rnn.WaBresolutions[i][0]*this.rnn.WaBresolutions[i][1]*4;
        }
        return result;
    }
    this.Import = function(data) {
        var idx = 0;
        var idxp = this.rcnn.layerResolutions[0][0]*this.rcnn.layerResolutions[0][1]*3;
        for (var i=0;i<this.rcnn.WaB.length;i++) {
            for (var j=0;j<this.rcnn.WaB[i];j++) {
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.rcnn.WaB[i][j]);
                this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RGBF, this.rcnn.layerResolutions[i][0], this.rcnn.layerResolutions[i][1], 0, this.gl.RGB, self.FLOAT, data.subarray(idx,idx+idxp));
                idx += idxp;
            }
        }
        for (var i=0;i<this.rnn.WaB.length;i++) {
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.rnn.WaB[i]);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, self.RGBAF, this.rnn.WaBresolutions[i][0], this.rnn.WaBresolutions[i][1], 0, this.gl.RGBA, self.FLOAT, data.subarray(idx,idx+this.rnn.WaBresolutions[i][0]*this.rnn.WaBresolutions[i][1]*4));
            idx += this.rnn.WaBresolutions[i][0]*this.rnn.WaBresolutions[i][1]*4;
        }
    }
}