var KhawnEngine = function() {
    var self = this;
    this.canvas = document.createElement('canvas');
    this.canvas.width = 1280;
    this.canvas.height = 720;
    this.gl = this.canvas.getContext('webgl2',{antialias: true, alpha: true, depth: true, premultipliedAlpha: false});
    this.gl.enable(this.gl.DEPTH_TEST);
    this.gl.depthFunc(this.gl.LEQUAL);
    this.gl.getExtension('EXT_color_buffer_float');
    this.gl.getExtension('OES_texture_float_linear');
    this.gl.enable(this.gl.CULL_FACE);
    this.gl.cullFace(this.gl.FRONT);
    // this.gl.frontFace(this.gl.CCW);
    this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
    // this.gl.disable(this.gl.DEPTH_TEST);
    // this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
    // this.gl.enable(this.gl.BLEND);
    this.version = '0.0.7b';
	this.RtoD = 180/Math.PI;
	this.DtoR = Math.PI/180;
	this.Tau = 2*Math.PI;
	this.TextDecoder = new TextDecoder();
	this.TextEncoder = new TextEncoder();
	this.TextureReadRenderBuffer = self.gl.createFramebuffer();
    this.Texture = function(data,dims) {
        var texture = self.gl.createTexture();
		this.Image = null;
		this.type = 0;
		this.channels = dims;
        self.gl.bindTexture(self.gl.TEXTURE_2D, texture);
		if (data instanceof Image || data instanceof HTMLCanvasElement) {
			self.gl.texImage2D(self.gl.TEXTURE_2D, 0, self.gl.RGBA, self.gl.RGBA, self.gl.UNSIGNED_BYTE, data);
			self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.LINEAR);
			self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.LINEAR);
			self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.REPEAT);
			self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.REPEAT);
			this.texture = texture;
			this.height = data.height;
			this.width = data.width;
			this.Image = data;
			this.type = 0;
			this.channels = 4;
		} else if (data instanceof Float32Array) {
			var res = self.calculateMostEfficientTextureSize(data.length/dims);
			var a = [self.gl.RED,self.gl.RG,self.gl.RGB,self.gl.RGBA][dims-1];
			var b = [self.gl.R32F,self.gl.RG32F,self.gl.RGB32F,self.gl.RGBA32F][dims-1];
			self.gl.texImage2D(self.gl.TEXTURE_2D, 0, b, res[0], res[1], 0, a, self.gl.FLOAT, data);
			self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
			self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.NEAREST);
			self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.REPEAT);
			self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.REPEAT);
			this.texture = texture;
			this.height = res[1];
			this.width = res[0];
			this.type = 1;
		} else if (data instanceof Int32Array) {
			var res = self.calculateMostEfficientTextureSize(data.length/dims);
			var a = [self.gl.RED_INTEGER,self.gl.RG_INTEGER,self.gl.RGB_INTEGER,self.gl.RGBA_INTEGER][dims-1];
			var b = [self.gl.R32I,self.gl.RG32I,self.gl.RGB32I,self.gl.RGBA32I][dims-1];
			self.gl.texImage2D(self.gl.TEXTURE_2D, 0, b, res[0], res[1], 0, a, self.gl.INT, data);
			self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
			self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.NEAREST);
			self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.REPEAT);
			self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.REPEAT);
			this.texture = texture;
			this.height = res[1];
			this.width = res[0];
			this.type = 2;
		}
		this.__proto__.toArrayBuffer = function() {
			var arr = [];
			arr.push(self.TextEncoder.encode("Texture\x01"));
			self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, self.TextureReadRenderBuffer);
			self.gl.framebufferTexture2D(self.gl.FRAMEBUFFER, self.gl.COLOR_ATTACHMENT0,self.gl.TEXTURE_2D, this.texture, 0);
			self.gl.readBuffer(self.gl.COLOR_ATTACHMENT0);
			if (this.type === 0) {
				var bfr = new Uint8Array(this.channels*this.width*this.height);
				self.gl.readPixels(0,0,this.width,this.height,self.gl.RGBA,self.gl.UNSIGNED_BYTE,bfr);
			} else if (this.type === 1) {
				var bfr = new Float32Array(this.channels*this.width*this.height);
				self.gl.readPixels(0,0,this.width,this.height,[self.gl.RED,self.gl.RG,self.gl.RGB,self.gl.RGBA][this.channels-1],self.gl.FLOAT,bfr);
			} else if (this.type === 2) {
				var bfr = new Int32Array(this.channels*this.width*this.height);
				self.gl.readPixels(0,0,this.width,this.height,[self.gl.RED_INTEGER,self.gl.RG_INTEGER,self.gl.RGB_INTEGER,self.gl.RGBA_INTEGER][this.channels-1],self.gl.INT,bfr);
			} else {
				var bfr = new Uint8Array(0);
			}
			bfr = pako.gzip(new Uint8Array(bfr.buffer));
			arr.push(new Uint8Array(new Uint32Array([bfr.buffer.byteLength,[6,4,5][this.type]]).buffer));
			arr.push(bfr);
			var length = 0;
			for (var i=0; i<arr.length; i++) {
				length += arr[i].length;
			}
			arr[1].set(new Uint8Array(new Uint32Array([length-(arr[0].length+arr[1].length)]).buffer));
			var bufr = new Uint8Array(length);
			var idx = 0;
			for (var i=0; i<arr.length; i++) {
				bufr.set(arr[i],idx);
				idx += arr[i].length;
			}
			return bufr;
		}
    }
    this.VertexShader = function(source) {
        var shader = self.gl.createShader(self.gl.VERTEX_SHADER);
        self.gl.shaderSource(shader, source);
        self.gl.compileShader(shader);
        if (!self.gl.getShaderParameter(shader, self.gl.COMPILE_STATUS)) {
            console.error(self.gl.getShaderInfoLog(shader));
        }
        this.shader = shader;
        this.source = source;
    }
    this.FragmentShader = function(source) {
        var shader = self.gl.createShader(self.gl.FRAGMENT_SHADER);
        self.gl.shaderSource(shader, source);
        self.gl.compileShader(shader);
        if (!self.gl.getShaderParameter(shader, self.gl.COMPILE_STATUS)) {
            console.error(self.gl.getShaderInfoLog(shader));
        }
        this.shader = shader;
        this.source = source;
    }
    this.Shader = function(vs,fs,uniforms) {
        this.program = self.gl.createProgram();
        this.vs = vs;
        this.fs = fs;
        self.gl.attachShader(this.program,this.vs.shader);
        self.gl.attachShader(this.program,this.fs.shader);
        self.gl.linkProgram(this.program);
        self.gl.useProgram(this.program);
        if (!self.gl.getProgramParameter(this.program, self.gl.LINK_STATUS)) {
            console.error(self.gl.getProgramInfoLog(this.program));
        }
        this.ViewMatrix = self.gl.getUniformLocation(this.program, 'uViewMatrix');
		this.VertexPoitions = self.gl.getUniformLocation(this.program, 'uVertexPoitions');
		this.VertexNormals = self.gl.getUniformLocation(this.program, 'uVertexNormals');
		this.VertexUVs = self.gl.getUniformLocation(this.program, 'uVertexUVs');
		this.LightPositions = self.gl.getUniformLocation(this.program, 'LightPositions');
		this.LightColors = self.gl.getUniformLocation(this.program, 'LightColors');
        this.Options = self.gl.getUniformLocation(this.program, 'uOptions');
		this.CamreaPosition = self.gl.getUniformLocation(this.program, 'uCamreaPosition');
		this.ReflectionCubemap = self.gl.getUniformLocation(this.program, 'uReflectionCubemap');
		this.ReflectionPosition = self.gl.getUniformLocation(this.program, 'uReflectionPosition');
        this.Uniforms = {};
        for (var i = 0; i < uniforms.length; i++) {
            if (uniforms[i].Value == undefined) {
				this.Uniforms[uniforms[i].Name] = {Location:self.gl.getUniformLocation(this.program, uniforms[i].Name),Type:uniforms[i].Type};
			} else {
				var value = uniforms[i].Value;
				var U = {Location:self.gl.getUniformLocation(this.program, uniforms[i].Name),Type:uniforms[i].Type};
				if (U.Type == "float") {
					if (value instanceof Array) {
						self.gl.uniform1fv(U.Location,value);
					} else {
						self.gl.uniform1fv(U.Location,[value]);
					}
				} else if (U.Type == "vec3") {
					self.gl.uniform3fv(U.Location,value);
				} else if (U.Type == "vec4") {
					self.gl.uniform4fv(U.Location,value);
				} else if (U.Type == "mat3") {
					self.gl.uniformMatrix3fv(U.Location,false,value.flat());
				} else if (U.Type == "mat4") {
					self.gl.uniformMatrix4fv(U.Location,false,value.flat());
				} else if (U.Type == "sampler2D" || U.Type == "isampler2D") {
					if (value instanceof Array) {
						self.gl.uniform1iv(U.Location,value);
					} else {
						self.gl.uniform1iv(U.Location,[value]);
					}
				} else if (U.Type == "samplerCube") {
					if (value instanceof Array) {
						self.gl.uniform1iv(U.Location,value);
					} else {
						self.gl.uniform1iv(U.Location,[value]);
					}
				}
			}
        }
        this.VertexDataIndexs = self.gl.getAttribLocation(this.program, 'aVertexDataIndexs');
		self.gl.enableVertexAttribArray(this.VertexDataIndexs);
		this.__proto__.toArrayBuffer = function() {
			var arr = [];
			arr.push(self.TextEncoder.encode("Shader\x00"));
			arr.push(new Uint8Array(8));
			if (this.vs) {
				arr.push(self.TextEncoder.encode("vs\x01"));
				var b = self.TextEncoder.encode(this.vs.source);
				arr.push(new Uint8Array(new Uint32Array([b.buffer.byteLength,2]).buffer));
				arr.push(b);
			}
			if (this.VertexDataIndexBuffer) {
				arr.push(self.TextEncoder.encode("fs\x01"));
				var b = self.TextEncoder.encode(this.fs.source);
				arr.push(new Uint8Array(new Uint32Array([b.buffer.byteLength,2]).buffer));
				arr.push(b);
			}
			var length = 0;
			for (var i=0; i<arr.length; i++) {
				length += arr[i].length;
			}
			arr[1].set(new Uint8Array(new Uint32Array([length-(arr[0].length+arr[1].length)]).buffer));
			var bufr = new Uint8Array(length);
			var idx = 0;
			for (var i=0; i<arr.length; i++) {
				bufr.set(arr[i],idx);
				idx += arr[i].length;
			}
			return bufr;
		}
    }
	this.VertexMapingVertexShader = new self.VertexShader(`#version 300 es
		const highp vec4 p[4] = vec4[4](vec4(1.0,1.0,0.0,1.0),vec4(-1.0,1.0,0.0,1.0),vec4(1.0,-1.0,0.0,1.0),vec4(-1.0,-1.0,0.0,1.0));
		out highp vec4 fragpos;
		void main(void) {
			fragpos = p[gl_VertexID];
			gl_Position = fragpos;
		}
	`);
	this.PostProcessingShader = function(fs,uniforms) {
        this.program = self.gl.createProgram();
        this.vs = self.VertexMapingVertexShader;
        this.fs = fs;
        self.gl.attachShader(this.program,this.vs.shader);
        self.gl.attachShader(this.program,this.fs.shader);
        self.gl.linkProgram(this.program);
        self.gl.useProgram(this.program);
        if (!self.gl.getProgramParameter(this.program, self.gl.LINK_STATUS)) {
            console.error(self.gl.getProgramInfoLog(this.program));
        }
        this.Uniforms = {};
        for (var i = 0; i < uniforms.length; i++) {
			if (uniforms[i].Value == undefined) {
				this.Uniforms[uniforms[i].Name] = {Location:self.gl.getUniformLocation(this.program, uniforms[i].Name),Type:uniforms[i].Type};
			} else {
				var value = uniforms[i].Value;
				var U = {Location:self.gl.getUniformLocation(this.program, uniforms[i].Name),Type:uniforms[i].Type};
				if (U.Type == "float") {
					if (value instanceof Array) {
						self.gl.uniform1fv(U.Location,value);
					} else {
						self.gl.uniform1fv(U.Location,[value]);
					}
				} else if (U.Type == "vec3") {
					self.gl.uniform3fv(U.Location,value);
				} else if (U.Type == "vec4") {
					self.gl.uniform4fv(U.Location,value);
				} else if (U.Type == "mat3") {
					self.gl.uniformMatrix3fv(U.Location,false,value.flat());
				} else if (U.Type == "mat4") {
					self.gl.uniformMatrix4fv(U.Location,false,value.flat());
				} else if (U.Type == "mat2x3") {
					self.gl.uniformMatrix2x3fv(U.Location,value.flat());
				} else if (U.Type == "sampler2D") {
					if (value instanceof Array) {
						self.gl.uniform1iv(U.Location,value);
					} else {
						self.gl.uniform1iv(U.Location,[value]);
					}
				} else if (U.Type == "isampler2D") {
					self.gl.uniform1i(U.Location,value);
				}
			}
        }
    }
	this.ReflectionProbePostProcessingShader = new self.PostProcessingShader(new self.FragmentShader(`#version 300 es
		out highp vec4 color;
		
		uniform highp sampler2D uColorTex;
		uniform highp sampler2D uPositionTex;
		
		void main(void) {
			highp ivec2 Pos = ivec2(gl_FragCoord.xy-0.5);
			color = vec4(texelFetch(uColorTex,Pos,0).xyz,texelFetch(uPositionTex,Pos,0).w);
		}
	`),[{Name:'uColorTex',Type:'sampler2D',Value:0},{Name:'uPositionTex',Type:'sampler2D',Value:1}]);
	this.MotionBlurPostProcessingShader = new self.PostProcessingShader(new self.FragmentShader(`#version 300 es
		out highp vec4 color;
		
		uniform highp sampler2D uColorTex;
		uniform highp mat3 uMotionMat;
		uniform highp vec2 uAspect;
		
		highp ivec2 wh;
		in highp vec4 fragpos;
		void main(void) {
			highp vec3 Pos0 = vec3(fragpos.xy*uAspect,1);
			highp vec3 Pos1 = uMotionMat*Pos0;
			highp vec3 Color;
			for (highp float i=0.0; i<1.0; i+=0.0625) {
				highp vec3 p = mix(Pos0,Pos1,i);
				Color += texture(uColorTex,((p.xy/(p.z*uAspect))+1.0)/2.0).rgb;
			}
			Color *= 0.0625;
			color = vec4(Color,1.0);
			// color = vec4(texture(uColorTex,((Pos1.xy/Pos1.z)+1.0)/2.0).rgb,1.0);
		}
	`),[{Name:'uColorTex',Type:'sampler2D',Value:0},{Name:'uMotionMat',Type:'mat3'},{Name:'uAspect',Type:'vec2'}]);
	this.DepthOfFeildProcessingShader = new self.PostProcessingShader(new self.FragmentShader(`#version 300 es
		out highp vec4 color;
		
		uniform highp sampler2D uColorTex;
		uniform highp sampler2D uPosTex;
		uniform highp float uFocalDistance;
		uniform highp float uArperture;
		
		in highp vec4 fragpos;
		void main(void) {
			highp ivec2 Pos = ivec2(gl_FragCoord.xy-0.5);
			highp vec2 xy = (fragpos.xy+1.0)*0.5;
			// color = vec4(texture(uColorTex,xy,5.0).xyz,1.0);
			color = vec4(texture(uColorTex,xy,log2(uArperture*abs((texelFetch(uPosTex,Pos,1).w-1.0)/uFocalDistance)+1.0)).xyz,1.0);
		}
	`),[{Name:'uColorTex',Type:'sampler2D',Value:0},{Name:'uPosTex',Type:'sampler2D',Value:1},{Name:'uFocalDistance',Type:'float'},{Name:'uArperture',Type:'float'}]);
	this.BloomPostProcessingShader = new self.PostProcessingShader(new self.FragmentShader(`#version 300 es
		out highp vec4 color;
		
		uniform highp sampler2D uColorTex;
		
		in highp vec4 fragpos;
		void main(void) {
			highp ivec2 Pos = ivec2(gl_FragCoord.xy-0.5);
			highp vec2 xy = (fragpos.xy+1.0)*0.5;
			highp vec3 bloom = max(texture(uColorTex,xy,1.0).xyz-1.0,vec3(0));
			for (highp float i=1.0; i<32.0; i++) {
				bloom += max(texture(uColorTex,xy,log2(i)).xyz-i,vec3(0));
			}
			color = vec4(texelFetch(uColorTex,Pos,0).xyz+bloom,1.0);
		}
	`),[{Name:'uColorTex',Type:'sampler2D',Value:0}]);
	this.VertexMapingShader = function(fs,uniforms) {
        this.program = self.gl.createProgram();
        this.vs = self.VertexMapingVertexShader;
        this.fs = fs;
        self.gl.attachShader(this.program,this.vs.shader);
        self.gl.attachShader(this.program,this.fs.shader);
        self.gl.linkProgram(this.program);
        self.gl.useProgram(this.program);
        if (!self.gl.getProgramParameter(this.program, self.gl.LINK_STATUS)) {
            console.error(self.gl.getProgramInfoLog(this.program));
        }
		this.notNormals = self.gl.getUniformLocation(this.program, "notNormals");
		this.BlendShapeWeight = self.gl.getUniformLocation(this.program, "uBlendShapeWeight");
        this.Uniforms = {};
        for (var i = 0; i < uniforms.length; i++) {
			if (uniforms[i].Value == undefined) {
				this.Uniforms[uniforms[i].Name] = {Location:self.gl.getUniformLocation(this.program, uniforms[i].Name),Type:uniforms[i].Type};
			} else {
				var value = uniforms[i].Value;
				var U = {Location:self.gl.getUniformLocation(this.program, uniforms[i].Name),Type:uniforms[i].Type};
				if (U.Type == "float") {
					if (value instanceof Array) {
						self.gl.uniform1fv(U.Location,value);
					} else {
						self.gl.uniform1fv(U.Location,[value]);
					}
				} else if (U.Type == "vec3") {
					self.gl.uniform3fv(U.Location,value);
				} else if (U.Type == "vec4") {
					self.gl.uniform4fv(U.Location,value);
				} else if (U.Type == "mat3") {
					self.gl.uniformMatrix3fv(U.Location,false,value.flat());
				} else if (U.Type == "mat4") {
					self.gl.uniformMatrix4fv(U.Location,false,value.flat());
				} else if (U.Type == "sampler2D") {
					if (value instanceof Array) {
						self.gl.uniform1iv(U.Location,value);
					} else {
						self.gl.uniform1iv(U.Location,[value]);
					}
				} else if (U.Type == "isampler2D") {
					self.gl.uniform1i(U.Location,value);
				}
			}
        }
    }
	this.MeshVertexMapingShader = new this.VertexMapingShader(new self.FragmentShader(`#version 300 es
		out highp vec4 VertexPosition;
		
		uniform bool notNormals;
		uniform highp mat3x4 uModelMatrix;
		uniform highp sampler2D uVertexPositions;
		uniform highp sampler2D uVertexNormals;
		
		void main(void) {
			highp ivec2 Pos = ivec2(gl_FragCoord.xy-0.5);
			if (notNormals) {
				VertexPosition = vec4(vec4(texelFetch(uVertexPositions,Pos,0).xyz,1.0)*uModelMatrix,1.0);
			} else {
				VertexPosition = vec4(normalize(vec4(texelFetch(uVertexPositions,Pos,0).xyz,0.0)*uModelMatrix),0.0);
			}
		}
	`),[{Name:'uModelMatrix',Type:'mat3x4'},{Name:'uVertexPositions',Type:'sampler2D',Value:0},{Name:'uVertexNormals',Type:'sampler2D',Value:1}]);
	this.SkinnedMeshVertexMapingShader = new this.VertexMapingShader(new self.FragmentShader(`#version 300 es
		out highp vec4 VertexPosition;
		
		uniform bool notNormals;
		uniform highp sampler2D uVertexPositions;
		uniform highp sampler2D uTransforms[3];
		uniform highp isampler2D uWeightIndexs;
		uniform highp sampler2D uWeights;
		uniform highp sampler2D uBlendShapeAdd;
		
		void main(void) {
			highp ivec2 Pos = ivec2(gl_FragCoord.xy-0.5);
			highp vec4 v = texelFetch(uVertexPositions,Pos,0);
			v.xyz += texelFetch(uBlendShapeAdd,Pos,0).xyz;
			highp vec4 VertexP = vec4(v.xyz,notNormals);
			if (!notNormals) {
				highp int Wtmp = textureSize(uWeights,0)[0];
				highp int VPidx = int(v.w);
				Pos = ivec2(VPidx%Wtmp,VPidx/Wtmp);
			}
			highp int Wid = textureSize(uTransforms[0],0)[0];
			highp ivec4 WeightI = ivec4(texelFetch(uWeightIndexs,Pos,0));
			highp vec4 Weights = texelFetch(uWeights,Pos,0);
			highp ivec2 tp = ivec2(WeightI[0]%Wid,WeightI[0]/Wid);
			highp vec3 Vertex = (VertexP*mat3x4(texelFetch(uTransforms[0],tp,0),texelFetch(uTransforms[1],tp,0),texelFetch(uTransforms[2],tp,0)))*Weights[0];
			tp = ivec2(WeightI[1]%Wid,WeightI[1]/Wid);
			Vertex += (VertexP*mat3x4(texelFetch(uTransforms[0],tp,0),texelFetch(uTransforms[1],tp,0),texelFetch(uTransforms[2],tp,0)))*Weights[1];
			tp = ivec2(WeightI[2]%Wid,WeightI[2]/Wid);
			Vertex += (VertexP*mat3x4(texelFetch(uTransforms[0],tp,0),texelFetch(uTransforms[1],tp,0),texelFetch(uTransforms[2],tp,0)))*Weights[2];
			tp = ivec2(WeightI[3]%Wid,WeightI[3]/Wid);
			Vertex += (VertexP*mat3x4(texelFetch(uTransforms[0],tp,0),texelFetch(uTransforms[1],tp,0),texelFetch(uTransforms[2],tp,0)))*Weights[3];
			if (notNormals) {
				VertexPosition = vec4(Vertex,1);
			} else {
				VertexPosition = vec4(normalize(Vertex),0);
			}
		}
	`),[{Name:'uVertexPositions',Type:'sampler2D',Value:0},{Name:'uTransforms',Type:'sampler2D',Value:[1,2,3]},{Name:'uWeightIndexs',Type:'isampler2D',Value:4},{Name:'uWeights',Type:'sampler2D',Value:5},{Name:'uBlendShapeAdd',Type:'sampler2D',Value:6}]);
	this.SkinnedMeshBlendShapeVertexMapingShader = new this.VertexMapingShader(new self.FragmentShader(`#version 300 es
		out highp vec4 VertexPosition;
		
		// uniform highp sampler2D uVertexPositions;
		uniform highp isampler2D uBlendShapeIndexs;
		uniform highp sampler2D uBlendShapes;
		uniform highp float uBlendShapeWeight;
		
		void main(void) {
			highp ivec2 Pos = ivec2(gl_FragCoord.xy-0.5);
			highp int idx = texelFetch(uBlendShapeIndexs,Pos,0)[0];
			if (idx >= 0) {
				highp int Wid = textureSize(uBlendShapes,0)[0];
				VertexPosition = vec4(texelFetch(uBlendShapes,ivec2(idx%Wid,idx/Wid),0).xyz,uBlendShapeWeight);
			} else {
				VertexPosition = vec4(0);
			}
		}
	`),[{Name:'uBlendShapeIndexs',Type:'isampler2D',Value:0},{Name:'uBlendShapes',Type:'sampler2D',Value:1}]);
	this.VertexMapingIndexBuffer = self.gl.createBuffer();
	self.gl.bindBuffer(self.gl.ELEMENT_ARRAY_BUFFER, this.VertexMapingIndexBuffer);
	self.gl.bufferData(self.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0,2,1,3,1,2]), self.gl.STATIC_DRAW);
    this.Material = function(shader,uf) {
        this.shader = shader;
        this.Values = uf;
    }
	this.Primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987,1993,1997,1999,2003,2011,2017,2027,2029,2039,2053,2063,2069,2081,2083,2087,2089,2099,2111,2113,2129,2131,2137,2141,2143,2153,2161,2179,2203,2207,2213,2221,2237,2239,2243,2251,2267,2269,2273,2281,2287,2293,2297,2309,2311,2333,2339,2341,2347,2351,2357,2371,2377,2381,2383,2389,2393,2399,2411,2417,2423,2437,2441,2447,2459,2467,2473,2477,2503,2521,2531,2539,2543,2549,2551,2557,2579,2591,2593,2609,2617,2621,2633,2647,2657,2659,2663,2671,2677,2683,2687,2689,2693,2699,2707,2711,2713,2719,2729,2731,2741,2749,2753,2767,2777,2789,2791,2797,2801,2803,2819,2833,2837,2843,2851,2857,2861,2879,2887,2897,2903,2909,2917,2927,2939,2953,2957,2963,2969,2971,2999,3001,3011,3019,3023,3037,3041,3049,3061,3067,3079,3083,3089,3109,3119,3121,3137,3163,3167,3169,3181,3187,3191,3203,3209,3217,3221,3229,3251,3253,3257,3259,3271,3299,3301,3307,3313,3319,3323,3329,3331,3343,3347,3359,3361,3371,3373,3389,3391,3407,3413,3433,3449,3457,3461,3463,3467,3469,3491,3499,3511,3517,3527,3529,3533,3539,3541,3547,3557,3559,3571,3581,3583,3593,3607,3613,3617,3623,3631,3637,3643,3659,3671,3673,3677,3691,3697,3701,3709,3719,3727,3733,3739,3761,3767,3769,3779,3793,3797,3803,3821,3823,3833,3847,3851,3853,3863,3877,3881,3889,3907,3911,3917,3919,3923,3929,3931,3943,3947,3967,3989,4001,4003,4007,4013,4019,4021,4027,4049,4051,4057,4073,4079,4091,4093,4099,4111,4127,4129,4133,4139,4153,4157,4159,4177,4201,4211,4217,4219,4229,4231,4241,4243,4253,4259,4261,4271,4273,4283,4289,4297,4327,4337,4339,4349,4357,4363,4373,4391,4397,4409,4421,4423,4441,4447,4451,4457,4463,4481,4483,4493,4507,4513,4517,4519,4523,4547,4549,4561,4567,4583,4591,4597,4603,4621,4637,4639,4643,4649,4651,4657,4663,4673,4679,4691,4703,4721,4723,4729,4733,4751,4759,4783,4787,4789,4793,4799,4801,4813,4817,4831,4861,4871,4877,4889,4903,4909,4919,4931,4933,4937,4943,4951,4957,4967,4969,4973,4987,4993,4999,5003,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087,5099,5101,5107,5113,5119,5147,5153,5167,5171,5179,5189,5197,5209,5227,5231,5233,5237,5261,5273,5279,5281,5297,5303,5309,5323,5333,5347,5351,5381,5387,5393,5399,5407,5413,5417,5419,5431,5437,5441,5443,5449,5471,5477,5479,5483,5501,5503,5507,5519,5521,5527,5531,5557,5563,5569,5573,5581,5591,5623,5639,5641,5647,5651,5653,5657,5659,5669,5683,5689,5693,5701,5711,5717,5737,5741,5743,5749,5779,5783,5791,5801,5807,5813,5821,5827,5839,5843,5849,5851,5857,5861,5867,5869,5879,5881,5897,5903,5923,5927,5939,5953,5981,5987,6007,6011,6029,6037,6043,6047,6053,6067,6073,6079,6089,6091,6101,6113,6121,6131,6133,6143,6151,6163,6173,6197,6199,6203,6211,6217,6221,6229,6247,6257,6263,6269,6271,6277,6287,6299,6301,6311,6317,6323,6329,6337,6343,6353,6359,6361,6367,6373,6379,6389,6397,6421,6427,6449,6451,6469,6473,6481,6491,6521,6529,6547,6551,6553,6563,6569,6571,6577,6581,6599,6607,6619,6637,6653,6659,6661,6673,6679,6689,6691,6701,6703,6709,6719,6733,6737,6761,6763,6779,6781,6791,6793,6803,6823,6827,6829,6833,6841,6857,6863,6869,6871,6883,6899,6907,6911,6917,6947,6949,6959,6961,6967,6971,6977,6983,6991,6997,7001,7013,7019,7027,7039,7043,7057,7069,7079,7103,7109,7121,7127,7129,7151,7159,7177,7187,7193,7207,7211,7213,7219,7229,7237,7243,7247,7253,7283,7297,7307,7309,7321,7331,7333,7349,7351,7369,7393,7411,7417,7433,7451,7457,7459,7477,7481,7487,7489,7499,7507,7517,7523,7529,7537,7541,7547,7549,7559,7561,7573,7577,7583,7589,7591,7603,7607,7621,7639,7643,7649,7669,7673,7681,7687,7691,7699,7703,7717,7723,7727,7741,7753,7757,7759,7789,7793,7817,7823,7829,7841,7853,7867,7873,7877,7879,7883,7901,7907,7919,7927,7933,7937,7949,7951,7963,7993,8009,8011,8017,8039,8053,8059,8069,8081,8087,8089,8093,8101,8111,8117,8123,8147,8161,8167,8171,8179,8191,8209,8219,8221,8231,8233,8237,8243,8263,8269,8273,8287,8291,8293,8297,8311,8317,8329,8353,8363,8369,8377,8387,8389,8419,8423,8429,8431,8443,8447,8461,8467,8501,8513,8521,8527,8537,8539,8543,8563,8573,8581,8597,8599,8609,8623,8627,8629,8641,8647,8663,8669,8677,8681,8689,8693,8699,8707,8713,8719,8731,8737,8741,8747,8753,8761,8779,8783,8803,8807,8819,8821,8831,8837,8839,8849,8861,8863,8867,8887,8893,8923,8929,8933,8941,8951,8963,8969,8971,8999,9001,9007,9011,9013,9029,9041,9043,9049,9059,9067,9091,9103,9109,9127,9133,9137,9151,9157,9161,9173,9181,9187,9199,9203,9209,9221,9227,9239,9241,9257,9277,9281,9283,9293,9311,9319,9323,9337,9341,9343,9349,9371,9377,9391,9397,9403,9413,9419,9421,9431,9433,9437,9439,9461,9463,9467,9473,9479,9491,9497,9511,9521,9533,9539,9547,9551,9587,9601,9613,9619,9623,9629,9631,9643,9649,9661,9677,9679,9689,9697,9719,9721,9733,9739,9743,9749,9767,9769,9781,9787,9791,9803,9811,9817,9829,9833,9839,9851,9857,9859,9871,9883,9887,9901,9907,9923,9929,9931,9941,9949,9967,9973,10007,10009,10037,10039,10061,10067,10069,10079,10091,10093,10099,10103,10111,10133,10139,10141,10151,10159,10163,10169,10177,10181,10193,10211,10223,10243,10247,10253,10259,10267,10271,10273,10289,10301,10303,10313,10321,10331,10333,10337,10343,10357,10369,10391,10399,10427,10429,10433,10453,10457,10459,10463,10477,10487,10499,10501,10513,10529,10531,10559,10567,10589,10597,10601,10607,10613,10627,10631,10639,10651,10657,10663,10667,10687,10691,10709,10711,10723,10729,10733,10739,10753,10771,10781,10789,10799,10831,10837,10847,10853,10859,10861,10867,10883,10889,10891,10903,10909,10937,10939,10949,10957,10973,10979,10987,10993,11003,11027,11047,11057,11059,11069,11071,11083,11087,11093,11113,11117,11119,11131,11149,11159,11161,11171,11173,11177,11197,11213,11239,11243,11251,11257,11261,11273,11279,11287,11299,11311,11317,11321,11329,11351,11353,11369,11383,11393,11399,11411,11423,11437,11443,11447,11467,11471,11483,11489,11491,11497,11503,11519,11527,11549,11551,11579,11587,11593,11597,11617,11621,11633,11657,11677,11681,11689,11699,11701,11717,11719,11731,11743,11777,11779,11783,11789,11801,11807,11813,11821,11827,11831,11833,11839,11863,11867,11887,11897,11903,11909,11923,11927,11933,11939,11941,11953,11959,11969,11971,11981,11987,12007,12011,12037,12041,12043,12049,12071,12073,12097,12101,12107,12109,12113,12119,12143,12149,12157,12161,12163,12197,12203,12211,12227,12239,12241,12251,12253,12263,12269,12277,12281,12289,12301,12323,12329,12343,12347,12373,12377,12379,12391,12401,12409,12413,12421,12433,12437,12451,12457,12473,12479,12487,12491,12497,12503,12511,12517,12527,12539,12541,12547,12553,12569,12577,12583,12589,12601,12611,12613,12619,12637,12641,12647,12653,12659,12671,12689,12697,12703,12713,12721,12739,12743,12757,12763,12781,12791,12799,12809,12821,12823,12829,12841,12853,12889,12893,12899,12907,12911,12917,12919,12923,12941,12953,12959,12967,12973,12979,12983,13001,13003,13007,13009,13033,13037,13043,13049,13063,13093,13099,13103,13109,13121,13127,13147,13151,13159,13163,13171,13177,13183,13187,13217,13219,13229,13241,13249,13259,13267,13291,13297,13309,13313,13327,13331,13337,13339,13367,13381,13397,13399,13411,13417,13421,13441,13451,13457,13463,13469,13477,13487,13499,13513,13523,13537,13553,13567,13577,13591,13597,13613,13619,13627,13633,13649,13669,13679,13681,13687,13691,13693,13697,13709,13711,13721,13723,13729,13751,13757,13759,13763,13781,13789,13799,13807,13829,13831,13841,13859,13873,13877,13879,13883,13901,13903,13907,13913,13921,13931,13933,13963,13967,13997,13999,14009,14011,14029,14033,14051,14057,14071,14081,14083,14087,14107,14143,14149,14153,14159,14173,14177,14197,14207,14221,14243,14249,14251,14281,14293,14303,14321,14323,14327,14341,14347,14369,14387,14389,14401,14407,14411,14419,14423,14431,14437,14447,14449,14461,14479,14489,14503,14519,14533,14537,14543,14549,14551,14557,14561,14563,14591,14593,14621,14627,14629,14633,14639,14653,14657,14669,14683,14699,14713,14717,14723,14731,14737,14741,14747,14753,14759,14767,14771,14779,14783,14797,14813,14821,14827,14831,14843,14851,14867,14869,14879,14887,14891,14897,14923,14929,14939,14947,14951,14957,14969,14983,15013,15017,15031,15053,15061,15073,15077,15083,15091,15101,15107,15121,15131,15137,15139,15149,15161,15173,15187,15193,15199,15217,15227,15233,15241,15259,15263,15269,15271,15277,15287,15289,15299,15307,15313,15319,15329,15331,15349,15359,15361,15373,15377,15383,15391,15401,15413,15427,15439,15443,15451,15461,15467,15473,15493,15497,15511,15527,15541,15551,15559,15569,15581,15583,15601,15607,15619,15629,15641,15643,15647,15649,15661,15667,15671,15679,15683,15727,15731,15733,15737,15739,15749,15761,15767,15773,15787,15791,15797,15803,15809,15817,15823,15859,15877,15881,15887,15889,15901,15907,15913,15919,15923,15937,15959,15971,15973,15991,16001,16007,16033,16057,16061,16063,16067,16069,16073,16087,16091,16097,16103,16111,16127,16139,16141,16183,16187,16189,16193,16217,16223,16229,16231,16249,16253,16267,16273,16301,16319,16333,16339,16349,16361,16363,16369,16381,16411,16417,16421,16427,16433,16447,16451,16453,16477,16481,16487,16493,16519,16529,16547,16553,16561,16567,16573,16603,16607,16619,16631,16633,16649,16651,16657,16661,16673,16691,16693,16699,16703,16729,16741,16747,16759,16763,16787,16811,16823,16829,16831,16843,16871,16879,16883,16889,16901,16903,16921,16927,16931,16937,16943,16963,16979,16981,16987,16993,17011,17021,17027,17029,17033,17041,17047,17053,17077,17093,17099,17107,17117,17123,17137,17159,17167,17183,17189,17191,17203,17207,17209,17231,17239,17257,17291,17293,17299,17317,17321,17327,17333,17341,17351,17359,17377,17383,17387,17389,17393,17401,17417,17419,17431,17443,17449,17467,17471,17477,17483,17489,17491,17497,17509,17519,17539,17551,17569,17573,17579,17581,17597,17599,17609,17623,17627,17657,17659,17669,17681,17683,17707,17713,17729,17737,17747,17749,17761,17783,17789,17791,17807,17827,17837,17839,17851,17863,17881,17891,17903,17909,17911,17921,17923,17929,17939,17957,17959,17971,17977,17981,17987,17989,18013,18041,18043,18047,18049,18059,18061,18077,18089,18097,18119,18121,18127,18131,18133,18143,18149,18169,18181,18191,18199,18211,18217,18223,18229,18233,18251,18253,18257,18269,18287,18289,18301,18307,18311,18313,18329,18341,18353,18367,18371,18379,18397,18401,18413,18427,18433,18439,18443,18451,18457,18461,18481,18493,18503,18517,18521,18523,18539,18541,18553,18583,18587,18593,18617,18637,18661,18671,18679,18691,18701,18713,18719,18731,18743,18749,18757,18773,18787,18793,18797,18803,18839,18859,18869,18899,18911,18913,18917,18919,18947,18959,18973,18979,19001,19009,19013,19031,19037,19051,19069,19073,19079,19081,19087,19121,19139,19141,19157,19163,19181,19183,19207,19211,19213,19219,19231,19237,19249,19259,19267,19273,19289,19301,19309,19319,19333,19373,19379,19381,19387,19391,19403,19417,19421,19423,19427,19429,19433,19441,19447,19457,19463,19469,19471,19477,19483,19489,19501,19507,19531,19541,19543,19553,19559,19571,19577,19583,19597,19603,19609,19661,19681,19687,19697,19699,19709,19717,19727,19739,19751,19753,19759,19763,19777,19793,19801,19813,19819,19841,19843,19853,19861,19867,19889,19891,19913,19919,19927,19937,19949,19961,19963,19973,19979,19991,19993,19997,20011,20021,20023,20029,20047,20051,20063,20071,20089,20101,20107,20113,20117,20123,20129,20143,20147,20149,20161,20173,20177,20183,20201,20219,20231,20233,20249,20261,20269,20287,20297,20323,20327,20333,20341,20347,20353,20357,20359,20369,20389,20393,20399,20407,20411,20431,20441,20443,20477,20479,20483,20507,20509,20521,20533,20543,20549,20551,20563,20593,20599,20611,20627,20639,20641,20663,20681,20693,20707,20717,20719,20731,20743,20747,20749,20753,20759,20771,20773,20789,20807,20809,20849,20857,20873,20879,20887,20897,20899,20903,20921,20929,20939,20947,20959,20963,20981,20983,21001,21011,21013,21017,21019,21023,21031,21059,21061,21067,21089,21101,21107,21121,21139,21143,21149,21157,21163,21169,21179,21187,21191,21193,21211,21221,21227,21247,21269,21277,21283,21313,21317,21319,21323,21341,21347,21377,21379,21383,21391,21397,21401,21407,21419,21433,21467,21481,21487,21491,21493,21499,21503,21517,21521,21523,21529,21557,21559,21563,21569,21577,21587,21589,21599,21601,21611,21613,21617,21647,21649,21661,21673,21683,21701,21713,21727,21737,21739,21751,21757,21767,21773,21787,21799,21803,21817,21821,21839,21841,21851,21859,21863,21871,21881,21893,21911,21929,21937,21943,21961,21977,21991,21997,22003,22013,22027,22031,22037,22039,22051,22063,22067,22073,22079,22091,22093,22109,22111,22123,22129,22133,22147,22153,22157,22159,22171,22189,22193,22229,22247,22259,22271,22273,22277,22279,22283,22291,22303,22307,22343,22349,22367,22369,22381,22391,22397,22409,22433,22441,22447,22453,22469,22481,22483,22501,22511,22531,22541,22543,22549,22567,22571,22573,22613,22619,22621,22637,22639,22643,22651,22669,22679,22691,22697,22699,22709,22717,22721,22727,22739,22741,22751,22769,22777,22783,22787,22807,22811,22817,22853,22859,22861,22871,22877,22901,22907,22921,22937,22943,22961,22963,22973,22993,23003,23011,23017,23021,23027,23029,23039,23041,23053,23057,23059,23063,23071,23081,23087,23099,23117,23131,23143,23159,23167,23173,23189,23197,23201,23203,23209,23227,23251,23269,23279,23291,23293,23297,23311,23321,23327,23333,23339,23357,23369,23371,23399,23417,23431,23447,23459,23473,23497,23509,23531,23537,23539,23549,23557,23561,23563,23567,23581,23593,23599,23603,23609,23623,23627,23629,23633,23663,23669,23671,23677,23687,23689,23719,23741,23743,23747,23753,23761,23767,23773,23789,23801,23813,23819,23827,23831,23833,23857,23869,23873,23879,23887,23893,23899,23909,23911,23917,23929,23957,23971,23977,23981,23993,24001,24007,24019,24023,24029,24043,24049,24061,24071,24077,24083,24091,24097,24103,24107,24109,24113,24121,24133,24137,24151,24169,24179,24181,24197,24203,24223,24229,24239,24247,24251,24281,24317,24329,24337,24359,24371,24373,24379,24391,24407,24413,24419,24421,24439,24443,24469,24473,24481,24499,24509,24517,24527,24533,24547,24551,24571,24593,24611,24623,24631,24659,24671,24677,24683,24691,24697,24709,24733,24749,24763,24767,24781,24793,24799,24809,24821,24841,24847,24851,24859,24877,24889,24907,24917,24919,24923,24943,24953,24967,24971,24977,24979,24989,25013,25031,25033,25037,25057,25073,25087,25097,25111,25117,25121,25127,25147,25153,25163,25169,25171,25183,25189,25219,25229,25237,25243,25247,25253,25261,25301,25303,25307,25309,25321,25339,25343,25349,25357,25367,25373,25391,25409,25411,25423,25439,25447,25453,25457,25463,25469,25471,25523,25537,25541,25561,25577,25579,25583,25589,25601,25603,25609,25621,25633,25639,25643,25657,25667,25673,25679,25693,25703,25717,25733,25741,25747,25759,25763,25771,25793,25799,25801,25819,25841,25847,25849,25867,25873,25889,25903,25913,25919,25931,25933,25939,25943,25951,25969,25981,25997,25999,26003,26017,26021,26029,26041,26053,26083,26099,26107,26111,26113,26119,26141,26153,26161,26171,26177,26183,26189,26203,26209,26227,26237,26249,26251,26261,26263,26267,26293,26297,26309,26317,26321,26339,26347,26357,26371,26387,26393,26399,26407,26417,26423,26431,26437,26449,26459,26479,26489,26497,26501,26513,26539,26557,26561,26573,26591,26597,26627,26633,26641,26647,26669,26681,26683,26687,26693,26699,26701,26711,26713,26717,26723,26729,26731,26737,26759,26777,26783,26801,26813,26821,26833,26839,26849,26861,26863,26879,26881,26891,26893,26903,26921,26927,26947,26951,26953,26959,26981,26987,26993,27011,27017,27031,27043,27059,27061,27067,27073,27077,27091,27103,27107,27109,27127,27143,27179,27191,27197,27211,27239,27241,27253,27259,27271,27277,27281,27283,27299,27329,27337,27361,27367,27397,27407,27409,27427,27431,27437,27449,27457,27479,27481,27487,27509,27527,27529,27539,27541,27551,27581,27583,27611,27617,27631,27647,27653,27673,27689,27691,27697,27701,27733,27737,27739,27743,27749,27751,27763,27767,27773,27779,27791,27793,27799,27803,27809,27817,27823,27827,27847,27851,27883,27893,27901,27917,27919,27941,27943,27947,27953,27961,27967,27983,27997,28001,28019,28027,28031,28051,28057,28069,28081,28087,28097,28099,28109,28111,28123,28151,28163,28181,28183,28201,28211,28219,28229,28277,28279,28283,28289,28297,28307,28309,28319,28349,28351,28387,28393,28403,28409,28411,28429,28433,28439,28447,28463,28477,28493,28499,28513,28517,28537,28541,28547,28549,28559,28571,28573,28579,28591,28597,28603,28607,28619,28621,28627,28631,28643,28649,28657,28661,28663,28669,28687,28697,28703,28711,28723,28729,28751,28753,28759,28771,28789,28793,28807,28813,28817,28837,28843,28859,28867,28871,28879,28901,28909,28921,28927,28933,28949,28961,28979,29009,29017,29021,29023,29027,29033,29059,29063,29077,29101,29123,29129,29131,29137,29147,29153,29167,29173,29179,29191,29201,29207,29209,29221,29231,29243,29251,29269,29287,29297,29303,29311,29327,29333,29339,29347,29363,29383,29387,29389,29399,29401,29411,29423,29429,29437,29443,29453,29473,29483,29501,29527,29531,29537,29567,29569,29573,29581,29587,29599,29611,29629,29633,29641,29663,29669,29671,29683,29717,29723,29741,29753,29759,29761,29789,29803,29819,29833,29837,29851,29863,29867,29873,29879,29881,29917,29921,29927,29947,29959,29983,29989,30011,30013,30029,30047,30059,30071,30089,30091,30097,30103,30109,30113,30119,30133,30137,30139,30161,30169,30181,30187,30197,30203,30211,30223,30241,30253,30259,30269,30271,30293,30307,30313,30319,30323,30341,30347,30367,30389,30391,30403,30427,30431,30449,30467,30469,30491,30493,30497,30509,30517,30529,30539,30553,30557,30559,30577,30593,30631,30637,30643,30649,30661,30671,30677,30689,30697,30703,30707,30713,30727,30757,30763,30773,30781,30803,30809,30817,30829,30839,30841,30851,30853,30859,30869,30871,30881,30893,30911,30931,30937,30941,30949,30971,30977,30983,31013,31019,31033,31039,31051,31063,31069,31079,31081,31091,31121,31123,31139,31147,31151,31153,31159,31177,31181,31183,31189,31193,31219,31223,31231,31237,31247,31249,31253,31259,31267,31271,31277,31307,31319,31321,31327,31333,31337,31357,31379,31387,31391,31393,31397,31469,31477,31481,31489,31511,31513,31517,31531,31541,31543,31547,31567,31573,31583,31601,31607,31627,31643,31649,31657,31663,31667,31687,31699,31721,31723,31727,31729,31741,31751,31769,31771,31793,31799,31817,31847,31849,31859,31873,31883,31891,31907,31957,31963,31973,31981,31991,32003,32009,32027,32029,32051,32057,32059,32063,32069,32077,32083,32089,32099,32117,32119,32141,32143,32159,32173,32183,32189,32191,32203,32213,32233,32237,32251,32257,32261,32297,32299,32303,32309,32321,32323,32327,32341,32353,32359,32363,32369,32371,32377,32381,32401,32411,32413,32423,32429,32441,32443,32467,32479,32491,32497,32503,32507,32531,32533,32537,32561,32563,32569,32573,32579,32587,32603,32609,32611,32621,32633,32647,32653,32687,32693,32707,32713,32717,32719,32749];
	this.calculateMostEfficientTextureSize = function(l) {
		var options = [];
		var factored = [];
		var val = l;
		for (var i=0; (i<self.Primes.length && self.Primes[i] <= l); i++) {
			factored[i] = 0;
			for (; val%self.Primes[i] == 0;) {
				val /= self.Primes[i];
				factored[i]++;
			}
		}
		var best = [1,1];
		var bool = false;
		for (var i=0; i<factored.length;) {
			if (factored[i] > 0) {
				best[+bool] *= self.Primes[i];
				bool = !bool;
				factored[i]--;
			} else {
				i++;
			}
		}
		return best;
	}
    this.Mesh = function(MeshData) {
        this.MeshData = MeshData;
        this.VertexPositions = [];
        this.VertexDataIndexs = [];
		this.VertexUVPositions = [];
		this.VertexNormalDirections = [];
		this.IndexBuffers = [];
		this.VertexPositionBuffer = null;
		this.VertexDataIndexBuffer = null;
		this.VertexUVPositionBuffer = null;
		this.VertexNormalBuffer = null;
		this.IndexBuffersData = [];
		this.l = [];
		//mshDta.push({VertexPositions:VertexPositions,VertexNormals:VertexNormals,VertexDataIndexs:VertexDataIndexs,VertexUVs:newUVs,Indices:newIndcs,Weights:newWeights,WeightIndices:newWeightIndices});
		//this.VertexRenderBuffers = self.gl.createFramebuffer();
		//this.VertexPositionTextures = mesh.VertexPositions.map(function(t){return new self.Texture(new Float32Array(t.height*t.width*3),3)});
		if (this.MeshData.VertexPositions[0] instanceof Float64Array) {
			this.VertexPositionBuffer = new Float32Array(new Float64Array(this.MeshData.VertexPositions[0].buffer));
			var VB = new self.Texture(this.VertexPositionBuffer,3);
		} else {
			if (this.MeshData.VertexPositions[0] instanceof Float32Array) {
				this.VertexPositionBuffer = new Float32Array(this.MeshData.VertexPositions[0].buffer);
				var VB = new self.Texture(this.VertexPositionBuffer,3);
			} else {
				this.VertexPositionBuffer = new Float32Array(this.MeshData.VertexPositions.flat());
				var VB = new self.Texture(this.VertexPositionBuffer,3);
			}
		}
		if (this.MeshData.VertexUVs) {
			if (this.MeshData.VertexUVs[0] instanceof Float64Array) {
				this.VertexUVPositionBuffers = new Float32Array(new Float64Array(this.MeshData.VertexUVs[0].buffer));
				var UV = new self.Texture(this.VertexUVPositionBuffers,2);
			} else {
				if (this.MeshData.VertexUVs[0] instanceof Float32Array) {
					this.VertexUVPositionBuffers = new Float32Array(this.MeshData.VertexUVs[0].buffer);
					var UV = new self.Texture(this.VertexUVPositionBuffers,2);
				} else {
					this.VertexUVPositionBuffers = new Float32Array(this.MeshData.VertexUVs.flat());
					var UV = new self.Texture(this.VertexUVPositionBuffers,2);
				}
			}
		} else {
			var UV = new self.Texture(new Float32Array(2).fill(-1),2);
		}
		if (this.MeshData.VertexNormals) {
			if (this.MeshData.VertexNormals[0] instanceof Float64Array) {
				this.VertexNormalBuffer = new Float32Array(new Float64Array(this.MeshData.VertexNormals[0].buffer));
				var NB = new self.Texture(this.VertexNormalBuffer,this.MeshData.VertexNormals[0].length);
			} else {
				if (this.MeshData.VertexNormals[0] instanceof Float32Array) {
					this.VertexNormalBuffer = new Float32Array(this.MeshData.VertexNormals[0].buffer);
					var NB = new self.Texture(this.VertexNormalBuffer,this.MeshData.VertexNormals[0].length);
				} else {
					this.VertexNormalBuffer = new Float32Array(this.MeshData.VertexNormals.flat());
					var NB = new self.Texture(this.VertexNormalBuffer,this.MeshData.VertexNormals[0].length);
				}
			}
		} else {
			var NB = new self.Texture(new Float32Array(3).fill(0),3);
		}
		var VIB = self.gl.createBuffer();
		self.gl.bindBuffer(self.gl.ARRAY_BUFFER, VIB);
		if (this.MeshData.VertexDataIndexs) {
			if (this.MeshData.VertexDataIndexs[0] instanceof Int32Array) {
				this.VertexDataIndexBuffer = new Int32Array(this.MeshData.VertexDataIndexs[0].buffer);
				self.gl.bufferData(self.gl.ARRAY_BUFFER, this.VertexDataIndexBuffer, self.gl.STATIC_DRAW);
			} else {
				this.VertexDataIndexBuffer = new Int32Array(this.MeshData.VertexDataIndexs.flat());
				self.gl.bufferData(self.gl.ARRAY_BUFFER, this.VertexDataIndexBuffer, self.gl.STATIC_DRAW);
			}
		} else {
			this.VertexDataIndexBuffer = new Int32Array(this.MeshData.VertexPositions.length*3).map(function(dc,i){return i%3 == 0 ? Math.floor(i/3) : 0});
			self.gl.bufferData(self.gl.ARRAY_BUFFER, this.VertexDataIndexBuffer, self.gl.STATIC_DRAW);
		}
		this.VertexPositions = VB;
		this.VertexDataIndexs = VIB;
		this.VertexUVPositions = UV;
		this.VertexNormalDirections = NB;
		//var AllIndices = [];
		for (var i = 0; i < this.MeshData.Indices.length; i++) {
			var IB = self.gl.createBuffer();
			self.gl.bindBuffer(self.gl.ELEMENT_ARRAY_BUFFER, IB);
			if (this.MeshData.Indices[i][0] instanceof Uint16Array) {
				var ar = new Uint16Array(this.MeshData.Indices[i][0].buffer);
				self.gl.bufferData(self.gl.ELEMENT_ARRAY_BUFFER, ar, self.gl.STATIC_DRAW);
				//AllIndices.push(Array.from(ar));
			} else {
				var ar = new Uint16Array(this.MeshData.Indices[i].flat());
				self.gl.bufferData(self.gl.ELEMENT_ARRAY_BUFFER, ar, self.gl.STATIC_DRAW);
				//AllIndices.push(ar);
			}
			this.IndexBuffersData.push(ar);
			this.IndexBuffers.push(IB);
			this.l.push(this.MeshData.Indices[i].length*3);
        }
		this.__proto__.toArrayBuffer = function() {
			var arr = [];
			arr.push(self.TextEncoder.encode("Mesh\x00"));
			arr.push(new Uint8Array(8));
			if (this.VertexPositionBuffer) {
				arr.push(self.TextEncoder.encode("VertexPositions\x01"));
				arr.push(new Uint8Array(new Uint32Array([this.VertexPositionBuffer.buffer.byteLength,0]).buffer));
				arr.push(new Uint8Array(this.VertexPositionBuffer.buffer));
			}
			if (this.VertexDataIndexBuffer) {
				arr.push(self.TextEncoder.encode("VertexDataIndexs\x01"));
				arr.push(new Uint8Array(new Uint32Array([this.VertexDataIndexBuffer.buffer.byteLength,0]).buffer));
				arr.push(new Uint8Array(this.VertexDataIndexBuffer.buffer));
			}
			if (this.VertexUVPositionBuffer) {
				arr.push(self.TextEncoder.encode("VertexUVPositions\x01"));
				arr.push(new Uint8Array(new Uint32Array([this.VertexUVPositionBuffer.buffer.byteLength,0]).buffer));
				arr.push(new Uint8Array(this.VertexUVPositionBuffer.buffer));
			}
			if (this.VertexNormalBuffer) {
				arr.push(self.TextEncoder.encode("VertexNormalDirections\x01"));
				arr.push(new Uint8Array(new Uint32Array([this.VertexNormalBuffer.buffer.byteLength,0]).buffer));
				arr.push(new Uint8Array(this.VertexNormalBuffer.buffer));
			}
			if (this.IndexBuffersData.length > 0) {
				arr.push(self.TextEncoder.encode("Indexs\x00"));
				var idxtmp = arr.length;
				arr.push(new Uint8Array(8));
				var len = 0;
				for (var i=0; i<this.IndexBuffersData.length; i++) {
					arr.push(self.TextEncoder.encode("Material\x01"));
					arr.push(new Uint32Array([this.IndexBuffersData[i].buffer.byteLength,7]));
					arr.push(this.IndexBuffersData[i]);
					len += this.IndexBuffersData[i].buffer.byteLength+17;
				}
				arr[idxtmp] = new Uint32Array([len,0]);
			}
			var length = 0;
			for (var i=2; i<arr.length; i++) {
				length += arr[i].buffer.byteLength;
			}
			arr[1] = new Uint32Array([length,0]);
			return self.State.concatArrayOfBuffers(arr);
			//this.VertexPositionBuffer = null;
			//this.VertexDataIndexBuffer = null;
			//this.VertexUVPositionBuffer = null;
			//this.VertexNormalBuffer = null;
			
		}
		//this.IndicesTexture = new self.Texture(new Int32Array(AllIndices.flat()),3);
    }
    this.SkinnedMesh = function(MeshData,Pose,Bones) {
        this.MeshData = MeshData;
		this.DefaultPose = Pose;
		this.Bones = Bones;
		this.IndexBuffers = [];
		this.l = [];
		//mshDta.push({VertexPositions:VertexPositions,VertexNormals:VertexNormals,VertexDataIndexs:VertexDataIndexs,VertexUVs:newUVs,Indices:newIndcs,Weights:newWeights,WeightIndices:newWeightIndices});
		//this.VertexRenderBuffers = self.gl.createFramebuffer();
		//this.VertexPositionTextures = mesh.VertexPositions.map(function(t){return new self.Texture(new Float32Array(t.height*t.width*3),3)});
		if (this.MeshData.VertexPositions[0] instanceof Float64Array) {
			var VB = new self.Texture(new Float32Array(new Float64Array(this.MeshData.VertexPositions[0].buffer)),3);
		} else {
			if (this.MeshData.VertexPositions[0] instanceof Float32Array) {
				var VB = new self.Texture(new Float32Array(this.MeshData.VertexPositions[0].buffer),3);
			} else {
				var VB = new self.Texture(new Float32Array(this.MeshData.VertexPositions.flat()),3)
			}
		}
		if (this.MeshData.VertexUVs) {
			if (this.MeshData.VertexUVs[0] instanceof Float64Array) {
				var UV = new self.Texture(new Float32Array(new Float64Array(this.MeshData.VertexUVs[0].buffer)),2);
			} else {
				if (this.MeshData.VertexUVs[0] instanceof Float32Array) {
					var UV = new self.Texture(new Float32Array(this.MeshData.VertexUVs[0].buffer),2);
				} else {
					var UV = new self.Texture(new Float32Array(this.MeshData.VertexUVs.flat()),2);
				}
			}
		} else {
			var UV = new self.Texture(new Float32Array(2).fill(-1),2);
		}
		if (this.MeshData.VertexNormals) {
			if (this.MeshData.VertexNormals[0] instanceof Float64Array) {
				var NB = new self.Texture(new Float32Array(new Float64Array(this.MeshData.VertexNormals[0].buffer)),4);
			} else {
				if (this.MeshData.VertexNormals[0] instanceof Float32Array) {
					var NB = new self.Texture(new Float32Array(this.MeshData.VertexNormals[0].buffer),4);
				} else {
					var NB = new self.Texture(new Float32Array(this.MeshData.VertexNormals.flat()),4);
				}
			}
			var NRT = new self.Texture(new Float32Array(this.MeshData.VertexNormals.length*4).fill(0),4);
		} else {
			var NB = new self.Texture(new Float32Array(4).fill(0),4);
			var NRT = new self.Texture(new Float32Array(4).fill(0),4);
		}
		var VIB = self.gl.createBuffer();
		self.gl.bindBuffer(self.gl.ARRAY_BUFFER, VIB);
		if (this.MeshData.VertexDataIndexs) {
			if (this.MeshData.VertexDataIndexs[0] instanceof Int32Array) {
				self.gl.bufferData(self.gl.ARRAY_BUFFER, new Int32Array(this.MeshData.VertexDataIndexs[0].buffer), self.gl.STATIC_DRAW);
			} else {
				self.gl.bufferData(self.gl.ARRAY_BUFFER, new Int32Array(this.MeshData.VertexDataIndexs.flat()), self.gl.STATIC_DRAW);
			}
		} else {
			var arr = new Int32Array(this.MeshData.VertexPositions.length*3).map(function(dc,i){return i%3 == 0 ? Math.floor(i/3) : 0});
			self.gl.bufferData(self.gl.ARRAY_BUFFER, arr, self.gl.STATIC_DRAW);
		}
		if (this.MeshData.Weights[0] instanceof Float32Array) {
			var WB = new self.Texture(new Float32Array(this.MeshData.Weights[0].buffer),4);
		} else {
			if (this.MeshData.Weights[0] instanceof Float64Array) {
				var WB = new self.Texture(new Float32Array(new Float64Array(this.MeshData.Weights[0].buffer)),4);
			} else {
				var WB = new self.Texture(new Float32Array(this.MeshData.Weights.flat()),4);
			}
		}
		if (this.MeshData.WeightIndexs[0] instanceof Int32Array) {
			var WIB = new self.Texture(new Int32Array(this.MeshData.WeightIndexs[0].buffer),4);
		} else {
			var WIB = new self.Texture(new Int32Array(this.MeshData.WeightIndexs.flat()),4);
		}
		var BLSP = [];
		if (this.MeshData.BlendShapes) {
			for (var i=0; i<this.MeshData.BlendShapes.length; i++) {
				var BLSPidxs = new Int32Array(this.MeshData.VertexPositions.length).fill(-1);
				for (var j=0; j<this.MeshData.BlendShapes[i].Indexs.length; j++) {
					BLSPidxs[this.MeshData.BlendShapes[i].Indexs[j]] = j;
				}
				var BLSPidxsTex = new self.Texture(BLSPidxs,1);
				if (this.MeshData.BlendShapes[i].BlendShapeNormals[0] instanceof Float64Array) {
					var BLSPnormsTex = new self.Texture(new Float32Array(new Float64Array(this.MeshData.BlendShapes[i].BlendShapeNormals[0].buffer)),3);
				} else if (this.MeshData.BlendShapes[i].BlendShapeNormals[0] instanceof Float32Array) {
					var BLSPnormsTex = new self.Texture(new Float32Array(this.MeshData.BlendShapes[i].BlendShapeNormals[0].buffer),3);
				} else {
					var BLSPnormsTex = new self.Texture(new Float32Array(this.MeshData.BlendShapes[i].BlendShapeNormals.flat()),3);
				}
				// if (BLSPnormsTex.height == 1 && BLSPnormsTex.width == 1) {
				// 	console.log(this.MeshData.BlendShapes[i].BlendShapeNormals.length);
				// }
				BLSP.push({Name:this.MeshData.BlendShapes[i].Name,BlendShapeNormals:BLSPnormsTex,Indexs:BLSPidxsTex});
			}
		}
		//NormalWeights NormalWeightIndexs
		this.VertexPositions = VB;
		this.VertexDataIndexs = VIB;
		this.VertexUVPositions = UV;
		this.VertexNormalDirections = NB;
		this.Weights = WB;
		this.WeightIndexs = WIB;
		this.BlendShapes = BLSP;
		//var AllIndices = [];
        for (var i = 0; i < this.MeshData.Indices.length; i++) {
			var IB = self.gl.createBuffer();
            self.gl.bindBuffer(self.gl.ELEMENT_ARRAY_BUFFER, IB);
            if (this.MeshData.Indices[i][0] instanceof Uint16Array) {
				var ar = new Uint16Array(this.MeshData.Indices[i][0].buffer);
                self.gl.bufferData(self.gl.ELEMENT_ARRAY_BUFFER, ar, self.gl.STATIC_DRAW);
				//AllIndices.push(Array.from(ar));
            } else {
				var ar = this.MeshData.Indices[i].flat();
                self.gl.bufferData(self.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(ar), self.gl.STATIC_DRAW);
				//AllIndices.push(ar);
            }
			this.IndexBuffers.push(IB);
			this.l.push(this.MeshData.Indices[i].length*3);
        }
		//this.IndicesTexture = new self.Texture(new Int32Array(AllIndices.flat()),3);
    }
    this.FBXdecoder = function() {
        this.parsed = {};
		this.MeshDatas = [];
		this.Heigerarchy = {};
        this.parse = function(FBXdata) {
            Error.stackTraceLimit = 128;
            var DataOut = {};
            function decodeProperty(dat) {
                var PropertyType = String.fromCharCode(dat[0]);
                if (PropertyType == "Y") {
                    return {leng:3,Value:new Int16Array(dat.slice(1,3).buffer)[0]};
                }
                if (PropertyType == "C") {
                    return {leng:2,Value:!!dat[1]};
                }
                if (PropertyType == "I") {
                    return {leng:5,Value:new Int32Array(dat.slice(1,5).buffer)[0]};
                }
                if (PropertyType == "F") {
                    return {leng:5,Value:new Float32Array(dat.slice(1,5).buffer)[0]};
                }
                if (PropertyType == "D") {
                    return {leng:9,Value:new Float64Array(dat.slice(1,9).buffer)[0]};
                }
                if (PropertyType == "L") {
                    return {leng:9,Value:"0x"+(new BigInt64Array(dat.slice(1,9).buffer)[0].toString(16))};
                    // return {leng:9,Value:new Int32Array(dat.slice(1,9).buffer)[0]};
                }
                if (PropertyType == "R") {
                    var Length = new Uint32Array(dat.slice(1,5).buffer)[0];
                    return {leng:5+Length,Value:dat.slice(5,5+Length)};
                }
                if (PropertyType == "S") {
                    var Length = new Uint32Array(dat.slice(1,5).buffer)[0];
                    return {leng:5+Length,Value:Array.from(dat.slice(5,5+Length)).map(function(xin){return String.fromCharCode(xin)}).join("")};
                }
                if (PropertyType == "f") {
                    var ArrLength = new Uint32Array(dat.slice(1,5).buffer)[0]*4;
                    var Encoding = new Uint32Array(dat.slice(5,9).buffer)[0];
                    var CompressedLength = new Uint32Array(dat.slice(9,13).buffer)[0];
                    if (Encoding == 0) {
                        return {leng:13+ArrLength,Value:new Float32Array(dat.slice(13,13+ArrLength).buffer)};
                    } else {
                        return {leng:13+CompressedLength,Value:new Float32Array(pako.ungzip(dat.slice(13,13+CompressedLength)).buffer)};
                    }
                }
                if (PropertyType == "d") {
                    var ArrLength = new Uint32Array(dat.slice(1,5).buffer)[0]*8;
                    var Encoding = new Uint32Array(dat.slice(5,9).buffer)[0];
                    var CompressedLength = new Uint32Array(dat.slice(9,13).buffer)[0];
                    if (Encoding == 0) {
                        return {leng:13+ArrLength,Value:new Float64Array(dat.slice(13,13+ArrLength).buffer)};
                    } else {
                        return {leng:13+CompressedLength,Value:new Float64Array(pako.ungzip(dat.slice(13,13+CompressedLength)).buffer)};
                    }
                }
                if (PropertyType == "l") {
                    var ArrLength = new Uint32Array(dat.slice(1,5).buffer)[0]*8;
                    var Encoding = new Uint32Array(dat.slice(5,9).buffer)[0];
                    var CompressedLength = new Uint32Array(dat.slice(9,13).buffer)[0];
                    if (Encoding == 0) {
                        return {leng:13+ArrLength,Value:new BigInt64Array(dat.slice(13,13+ArrLength).buffer)};
                    } else {
                        return {leng:13+CompressedLength,Value:new BigInt64Array(pako.ungzip(dat.slice(13,13+CompressedLength)).buffer)};
                    }
                }
                if (PropertyType == "i") {
                    var ArrLength = new Uint32Array(dat.slice(1,5).buffer)[0]*4;
                    var Encoding = new Uint32Array(dat.slice(5,9).buffer)[0];
                    var CompressedLength = new Uint32Array(dat.slice(9,13).buffer)[0];
                    if (Encoding == 0) {
                        return {leng:13+ArrLength,Value:new Int32Array(dat.slice(13,13+ArrLength).buffer)};
                    } else {
                        return {leng:13+CompressedLength,Value:new Int32Array(pako.ungzip(dat.slice(13,13+CompressedLength)).buffer)};
                    }
                }
                if (PropertyType == "b") {
                    var ArrLength = new Uint32Array(dat.slice(1,5).buffer)[0];
                    var Encoding = new Uint32Array(dat.slice(5,9).buffer)[0];
                    var CompressedLength = new Uint32Array(dat.slice(9,13).buffer)[0];
                    if (Encoding == 0) {
                        return {leng:13+ArrLength,Value:Array.from(dat.slice(13,13+ArrLength)).map(function(val){return !!val})};
                    } else {
                        return {leng:13+CompressedLength,Value:Array.from(pako.ungzip(dat.slice(13,13+CompressedLength))).map(function(val){return !!val})};
                    }
                }
                
                console.error("FBX Unknown PropertyType: "+PropertyType,dat+"\nYour FBX kinda sus");
            }
            function decodeNode(U8,idx) {
                if (U8.slice(0,13).reduce(function(a,b){return a+b}) == 0) {
                    return {leng:13};
                }
                var EndOffset = new Uint32Array(U8.slice(0,4).buffer)[0];
                var NumProperties = new Uint32Array(U8.slice(4,8).buffer)[0];
                var PropertyListLen = new Uint32Array(U8.slice(8,12).buffer)[0];
                var NameLen = U8[12];
                var Name = Array.from(U8.slice(13,13+NameLen)).map(function(v) {return String.fromCharCode(v);}).join("");
                var i = 13+NameLen;
                // if (U8.slice(i,i+13).reduce(function(a,b){return a+b}) == 0) {
                //     return {leng:13};
                // }
                var PropertyList = [];
                // PropertyListLen--;
                // console.log(idx,Name,NumProperties,PropertyListLen,EndOffset);
                for (var j=0; j<NumProperties; j++) {
                    var Property = decodeProperty(U8.subarray(i));
                    PropertyList.push(Property.Value);
                    i += Property.leng;
                }
                var result = {Name:Name,Content:{Properties:PropertyList}};
                var l = EndOffset-idx;
                for (; i<l;) {
                    var CurrentNode = decodeNode(U8.subarray(i),i+idx);
                    if (result.Content[CurrentNode.Name]) {
                        result.Content[CurrentNode.Name].push(CurrentNode.Content);
                    } else {
                        result.Content[CurrentNode.Name] = [CurrentNode.Content];
                    }
                    i += CurrentNode.leng;
                }
                result.leng = l;
                return result;
            }
            for (var i=27; i<FBXdata.length;) {
                if (FBXdata.slice(i,i+13).reduce(function(a,b){return a+b}) !== 0) {
                    var Curr = decodeNode(FBXdata.subarray(i),i);
                    if (DataOut[Curr.Name]) {
                        DataOut[Curr.Name].push(Curr.Content);
                    } else {
                        DataOut[Curr.Name] = [Curr.Content];
                    }
                    i += Curr.leng;
                } else {
                    break;
                }
            }
            // DataOut.version = new Uint32Array(new Uint8Array(Array.from(FBXdata.slice(23,26)).concat([0])).buffer)[0];
            this.parsed = DataOut;
        }
        this.decodeData = function() {
            var Nodes = {"0x0":{Children:[]}};
            var Objects = this.parsed.Objects;
            for (var i=0; i<Objects.length; i++) {
                var keys = Object.keys(Objects[i]);
                for (var j=0; j<keys.length-1; j++) {
                    var key = keys[j];
                    var Vals = Objects[i][key];
					if (key == "Pose") {
						var Poses = Objects[i][key];
						for (var k=0; k<Poses.length; k++) {
							var PoseNodes = Poses[k].PoseNode;
							for (var n=0; n<PoseNodes.length; n++) {
								var NodeID = PoseNodes[n].Node[0].Properties[0];
								if (!Nodes[NodeID]) {
									Nodes[NodeID] = {Matrix:PoseNodes[n].Matrix};
								} else {
									Nodes[NodeID].Matrix = PoseNodes[n].Matrix;
								}
							}
						}
					} else {
						for (var k=0; k<Vals.length-1; k++) {
							var Node = Vals[k];
							var NodeID = Node.Properties[0];
							Node.Children = [];
							if (!Nodes[NodeID]) {
								Nodes[NodeID] = Node;
							} else {
								Object.assign(Nodes[NodeID],Node);
							}
						}
					}
                }
            }
            var Connections = this.parsed.Connections[0].C;
            for (var i=0; i<Connections.length; i++) {
                var Connection = Connections[i];
                if (Connection.Properties[0] == "OO") {
                    var ParrentID = Connection.Properties[2];
                    var ChildID = Connection.Properties[1];
                    if (Nodes[ParrentID]) {
                        // if (Nodes[ChildID]) {
                        //     Nodes[ChildID].Parent = Nodes[ParrentID];
                        // }
                        Nodes[ParrentID].Children.push(Nodes[ChildID]);
                    }
                }
            }
            this.Heigerarchy = Nodes["0x0"];
        }
		this.ParseMesh = function(N) {
			var mshDta = {Indices:[]};
			var proptyps = N.Children[0].Children.map(function(nd){return nd.Properties[2]});
			var wIdx = proptyps.indexOf("Skin");
			var bIdx = proptyps.indexOf("BlendShape");
			var isSkinned = (wIdx !== -1);
			var hasBlendShapes =  (bIdx !== -1);
			var Bones = [];
			var BoneNodeIDs = [];
			for (var i=0; i<N.Children.length; i++) {
				var Child = N.Children[i];
				if (Child && Child.Vertices) {
					var VertexPositionBuffer = Child.Vertices[0].Properties[0];
					var VertexPositions = [];
					for (var j = 0; j < VertexPositionBuffer.length; j+=3) {
						VertexPositions.push(VertexPositionBuffer.subarray(j,j+3));
					}
					var indcs = Child.PolygonVertexIndex[0].Properties[0];
					if (Child.LayerElementUV) {
						var UVs = Child.LayerElementUV[0].UV[0].Properties[0];
						var UVIdxs = Child.LayerElementUV[0].UVIndex[0].Properties[0];
					} else {
						var UVs = new Float64Array(2);
						var UVIdxs = new Int32Array(0);
					}
					var Normals = Child.LayerElementNormal[0].Normals[0].Properties[0];
					var newUVs = [];
					for (var j = 0; j < UVs.length; j+=2) {
						UVs[j+1] = 1-UVs[j+1];
						newUVs.push(UVs.subarray(j,j+2));
					}
					var NormalIdxs = [];
					var newNorms = [];
					var NormLookup = {};
					for (var j=0; j<Normals.length; j+=3) {
						//var n = Array.from(Normals.subarray(j,j+3));
						var n = [Normals[j],Normals[j+1],Normals[j+2]];
						var idx = NormLookup[n];
						if (!idx) {
							newNorms.push([n[0],n[1],n[2],0]);
							idx = newNorms.length-1;
							NormLookup[n] = idx;
						}
						NormalIdxs.push(idx);
					}
					delete NormLookup;
					var NormBuffer = new Float32Array(newNorms.flat());
					var newNormals = [];
					for (var j = 0; j < NormBuffer.length; j+=4) {
						newNormals.push(NormBuffer.subarray(j,j+4));
					}
					var newIndcsBuffr = [];
					var newIndcs2 = [];
					var vertIdxs = [];
					var vartIdxDataLookup = {};
					for (var j = 0; j < indcs.length; j++) {
						var v = indcs[j];
						v = (v < 0) ? ((-v)-1) : v;
						var u = UVIdxs[j];
						var n = NormalIdxs[j];
						var vtx = [v,n,u];
						var vtxidx = vartIdxDataLookup[vtx];
						if (vtxidx) {
							newIndcs2.push(vtxidx);
						} else {
							newNormals[n][3] = v;
							newIndcs2.push(vertIdxs.length);
							vartIdxDataLookup[vtx] = vertIdxs.length;
							vertIdxs.push(vtx);
						}
					}
					delete vartIdxDataLookup;
					var newIndcs = [];
					for (var j = 0; j < indcs.length;) {
						for (var k = 1; true; k++) {
							if (indcs[j+k+1] >= 0) {
								newIndcsBuffr.push(newIndcs2[j]);
								newIndcsBuffr.push(newIndcs2[j+k]);
								newIndcsBuffr.push(newIndcs2[j+k+1]);
							} else {
								newIndcsBuffr.push(newIndcs2[j]);
								newIndcsBuffr.push(newIndcs2[j+k]);
								newIndcsBuffr.push(newIndcs2[j+k+1]);
								j += k+2;
								break;
							}
						}
					}
					newIndcsBuffr = new Uint16Array(newIndcsBuffr);
					for (var j = 0; j < newIndcsBuffr.length; j+=3) {
						newIndcs.push(newIndcsBuffr.subarray(j,j+3));
					}
					var VertexDataIndexBuffer = new Int32Array(vertIdxs.flat());
					var VertexDataIndexs = [];
					for (var j = 0; j < VertexDataIndexBuffer.length; j+=3) {
						VertexDataIndexs.push(VertexDataIndexBuffer.subarray(j,j+3));
					}
					if (isSkinned) {
						var BlendShapes = [];
						if (hasBlendShapes) {
							for (var j = 0; j < Child.Children[bIdx].Children.length; j++) {
								if (Child.Children[bIdx].Children[j]) {
									var Nme = Child.Children[bIdx].Children[j].Properties[1].split("\x00")[0];
									//Nme = Encoding.convert(Nme, {from:'SJIS'});
									var DataChild = Child.Children[bIdx].Children[j].Children[0];
									if (DataChild) {
										if (DataChild.Indexes[0].Properties[0].length) {
											var BlndShpeNormals = [];
											for (var k=0; k<DataChild.Vertices[0].Properties[0].length; k+=3) {
												BlndShpeNormals.push(DataChild.Vertices[0].Properties[0].subarray(k,k+3));
											}
											BlendShapes.push({Name:Nme,BlendShapeNormals:BlndShpeNormals,Indexs:DataChild.Indexes[0].Properties[0]});
										}
									}
								}
							}
							console.log(BlendShapes);
						}

						var WeightData = [];
						var idx = 0;
						for (var j = 0; j < Child.Children[wIdx].Children.length; j++) {
							if (Child.Children[wIdx].Children[j]) {
								// Encoding.convert(u8arr.slice(0,14), {from:'SJIS',type:'string'})
								var Nme = Child.Children[wIdx].Children[j].Properties[1].split("\x00")[0];
								// Nme = Encoding.convert(Nme, {from:'SJIS',type:'string'});
								Bones.push(Nme);
								BoneNodeIDs.push(Child.Children[wIdx].Children[j].Children[0].Properties[0]);
								if (Child.Children[wIdx].Children[j].Indexes) {
									var WeightIndices = Child.Children[wIdx].Children[j].Indexes[0].Properties[0];
									var weights = Child.Children[wIdx].Children[j].Weights[0].Properties[0];
									for (var k = 0; k < WeightIndices.length; k++) {
										if (WeightData[WeightIndices[k]]) {
											WeightData[WeightIndices[k]].push({Weight:weights[k],Index:idx});
										} else {
											WeightData[WeightIndices[k]] = [{Weight:weights[k],Index:idx}];
										}
									}
								}
								idx++;
							}
						}
						var newWeights = [];
						var newWeightIndices = [];
						var WeightsBuffer = new Float32Array(WeightData.length*4).fill(0);
						//var WeightIndicesBuffer = new Float32Array(WeightData.length*4).fill(0);
						var WeightIndicesBuffer = new Int32Array(WeightData.length*4).fill(0);
						var empty = [{Index:0,Weight:0},{Index:0,Weight:0},{Index:0,Weight:0},{Index:0,Weight:0}];
						for (var j = 0; j < WeightData.length; j++) {
							WeightData[j] = WeightData[j].sort(function(a,b) {return b.Weight - a.Weight;}).concat(empty).slice(0,4);
							var sum = WeightData[j][0].Weight + WeightData[j][1].Weight + WeightData[j][2].Weight + WeightData[j][3].Weight;
							WeightIndicesBuffer.set([WeightData[j][0].Index,WeightData[j][1].Index,WeightData[j][2].Index,WeightData[j][3].Index],j*4);
							WeightsBuffer.set([WeightData[j][0].Weight/sum,WeightData[j][1].Weight/sum,WeightData[j][2].Weight/sum,WeightData[j][3].Weight/sum],j*4);
							newWeights.push(WeightsBuffer.subarray(j*4,j*4+4));
							newWeightIndices.push(WeightIndicesBuffer.subarray(j*4,j*4+4));
						}
						//NormalWeights NormalWeightIndexs
						//mshDta.push({VertexPositions:VertexPositions,VertexDataIndexs:VertexDataIndexs,VertexNormals:newNormals,VertexUVs:newUVs,Indices:newIndcs,NormalWeights:newNormalWeights,NormalWeightIndexs:newNormalWeightIndexs,Weights:newWeights,WeightIndexs:newWeightIndices});

						mshDta.VertexPositions = VertexPositions;
						mshDta.VertexDataIndexs = VertexDataIndexs;
						mshDta.VertexNormals = newNormals;
						mshDta.VertexUVs = newUVs;
						mshDta.Weights = newWeights;
						mshDta.WeightIndexs = newWeightIndices;
						mshDta.BlendShapes = BlendShapes;
						mshDta.Indices.push(newIndcs);
						//mshDta.push({VertexPositions:VertexPositions,VertexDataIndexs:VertexDataIndexs,VertexNormals:newNormals,VertexUVs:newUVs,Indices:newIndcs,Weights:newWeights,WeightIndexs:newWeightIndices});
					} else {
						//mshDta.push({VertexPositions:VertexPositions,VertexNormals:newNormals,VertexDataIndexs:VertexDataIndexs,VertexUVs:newUVs,Indices:newIndcs});
						mshDta.VertexPositions = VertexPositions;
						mshDta.VertexDataIndexs = VertexDataIndexs;
						mshDta.VertexNormals = newNormals;
						mshDta.VertexUVs = newUVs;
						mshDta.Indices.push(newIndcs);
					}
				}
			}
			if (isSkinned) {
				return {MeshData:mshDta,Bones:Bones,isSkinned:true,BoneNodeIDs:BoneNodeIDs};
				//return {MeshData:mshDta,isSkinned:false};
			} else {
				return {MeshData:mshDta,isSkinned:false};
			}
		}
		this.toMeshDataArray = function() {
			this.MeshDatas = [];
			var dis = this;
            function findAndAddMeshes(nde) {
				for (var i=0; i<nde.Children.length; i++) {
                    if (nde.Children[i]) {
                        var nt = nde.Children[i].Properties[1].split("\x00\x01");
                        var Name = nt[0];
                        var Type0 = nt[1];
                        var Type1 = nde.Children[i].Properties[2];
						var nodeID = nde.Children[i].Properties[0];
                        if (Type1 == "Mesh") {
                            dis.MeshDatas.push({Data:dis.ParseMesh(nde.Children[i]),nodeID:nodeID});
                        } else {
							findAndAddMeshes(nde.Children[i]);
						}
                    }
                }
			}
			findAndAddMeshes(this.Heigerarchy);
			return this.MeshDatas.map(function(ms){return ms.Data});
		}
        this.toObject = function(Mat) {
			this.toMeshDataArray();
			var NodeIDObjectLookup = {};
			var dis = this;
            function makeChildren(nde,prt,trsfm) {
                for (var i=0; i<nde.Children.length; i++) {
                    if (nde.Children[i]) {
                        var nt = nde.Children[i].Properties[1].split("\x00\x01");
                        var Name = nt[0];
                        var Type0 = nt[1];
                        var Type1 = nde.Children[i].Properties[2];
						var nodeID = nde.Children[i].Properties[0];
						var NodeTransform = self.IdentityTransform;
						if (nde.Children[i].Matrix) {
							var M = nde.Children[i].Matrix[0].Properties[0];
							var Mat = new self.Matrix3x3([[M[0],M[1],M[2]],[M[4],M[5],M[6]],[M[8],M[9],M[10]]]).Transpose().ScaleThisBy(0.01);
							var Pos = new self.Vector3([M[12],M[13],M[14]]).ScaleThisBy(0.01);
							NodeTransform = new self.transform(Pos,Mat);
						}
                        if (Type1 == "Mesh") {
                            var newobj = new self.Object();
							newobj.Name = Name;
							prt.addChild(newobj);
							newobj.transform = trsfm.InverseTransformTransform(NodeTransform);
							NodeIDObjectLookup[nodeID] = newobj;
                        } else {
                           if (Type1 !== "Null") {
                                if (Type0 !== "NodeAttribute") {
                                    var newobj = new self.Object();
                                    newobj.Name = Name;
									NodeIDObjectLookup[nodeID] = newobj;
									newobj.transform = trsfm.InverseTransformTransform(NodeTransform);
									prt.addChild(makeChildren(nde.Children[i],newobj,NodeTransform));
                                } else {
									NodeIDObjectLookup[nodeID] = prt;
								}
                            } else {
								NodeIDObjectLookup[nodeID] = prt;
                                makeChildren(nde.Children[i],prt,trsfm.TransformTransform(NodeTransform));
                            }
                        }
                    }
                }
                return prt;
            }
            var Obj = makeChildren(this.Heigerarchy,new self.Object(),self.IdentityTransform);
            for (var i=0; i<this.MeshDatas.length; i++) {
                if (this.MeshDatas[i].Data.isSkinned) {
					var DefaultPose = [];
                    for (var j=0; j<this.MeshDatas[i].Data.BoneNodeIDs.length; j++) {
						//console.log(this.MeshDatas[i].Data.BoneNodeIDs[j],NodeIDObjectLookup[this.MeshDatas[i].Data.BoneNodeIDs[j]]);
						DefaultPose[j] = NodeIDObjectLookup[this.MeshDatas[i].Data.BoneNodeIDs[j]].getGlobalTransform();
                    }
                    var Mesh = new self.SkinnedMesh(this.MeshDatas[i].Data.MeshData,DefaultPose,this.MeshDatas[i].Data.Bones);
                    var Component = new self.Components.SkinnedMeshRenderer(Mesh,new Array(this.MeshDatas[i].Data.MeshData.Indices.length).fill(Mat || self.DefaultMaterial));
                    for (var j=0; j<this.MeshDatas[i].Data.Bones.length; j++) {
						Component.SetBoneRefrence(j,NodeIDObjectLookup[this.MeshDatas[i].Data.BoneNodeIDs[j]]);
						//var DebugBone = new self.CreateMeshObject(self.DebugBoneMesh,[self.DefaultMaterial]);
						//NodeIDObjectLookup[this.MeshDatas[i].Data.BoneNodeIDs[j]].addChild(DebugBone);
                    }
					//MeshDatas[i].Object.Name = "Skinned Mesh";
					//Component.Enabled = false;
					//nodeID
                    NodeIDObjectLookup[this.MeshDatas[i].nodeID].AddComponent(Component);
                } else {
                    var Mesh = new self.Mesh(this.MeshDatas[i].Data.MeshData);
                    var Component = new self.Components.MeshRenderer(Mesh,new Array(this.MeshDatas[i].Data.MeshData.Indices.length).fill(Mat || self.DefaultMaterial));
                    NodeIDObjectLookup[this.MeshDatas[i].nodeID].AddComponent(Component);
				}
            }
            // console.log(LimbNodes);
            Obj.setMatrix([[1,0,0],[0,0,1],[0,-1,0]]);
            return Obj;
        }
    }
	this.OBJdecoder = function() {
		this.parsed = {};
		this.decoded = {};
		this.parse = function(OBJdata) {
			OBJdata = self.TextDecoder.decode(OBJdata).split("\n");
			for (var i=0; i<OBJdata.length; i++) {
				OBJdata[i] = OBJdata[i].replaceAll("  "," ").replaceAll("  "," ").replaceAll("  "," ");
				if (!OBJdata[i]) {
					OBJdata.splice(i,1);
					i--;
				}
			}
			this.parsed = OBJdata;
		}
		this.decodeData = function() {
			this.decoded = {VertexPositions:[],VertexNormals:[],VertexUVs:[],VertexDataIndexs:[],Indices:[[]]};
			var IdxLookup = {};
			var Mtrl = -1;
			for (var i=0; i<this.parsed.length; i++) {
				var cur = this.parsed[i].split(" ");
				var type = cur.splice(0,1)[0];
				if (type == "v") {
					this.decoded.VertexPositions.push([parseFloat(cur[0]),parseFloat(cur[1]),parseFloat(cur[2])]);
				} else if (type == "vt") {
					this.decoded.VertexUVs.push([parseFloat(cur[0]),parseFloat(cur[1])]);
				} else if (type == "vn") {
					cur = [parseFloat(cur[0]) || 0,parseFloat(cur[1]) || 0,parseFloat(cur[2]) || 0];
					var l = Math.hypot(cur[0],cur[1],cur[2]);
					cur[0] /= l;
					cur[1] /= l;
					cur[2] /= l;
					this.decoded.VertexNormals.push(cur);
				} else if (type == "f") {
					for (var j=0; j<cur.length; j++) {
						cur[j] = cur[j].split("/");
						cur[j] = [parseInt(cur[j][0])-1,parseInt(cur[j][2])-1 || 0,parseInt(cur[j][1])-1 || 0];
						if (isNaN(cur[j][0])) {
							cur.splice(j,1);
							j--;
						} else {
							if (IdxLookup[cur[j]]) {
								cur[j] = IdxLookup[cur[j]];
							} else {
								IdxLookup[cur[j]] = this.decoded.VertexDataIndexs.length;
								this.decoded.VertexDataIndexs.push(cur[j]);
								cur[j] = this.decoded.VertexDataIndexs.length-1;
							}
						}
					}
					for (var j=2; j<cur.length; j++) {
						this.decoded.Indices[Mtrl == -1 ? 0 : Mtrl].push([cur[0],cur[j-1],cur[j]]);
						//this.decoded.Indices[Mtrl == -1 ? 0 : Mtrl].push([cur[j-(j%3)],cur[j-1],cur[j]]);
					}
				} else if (type == "usemtl") {
					Mtrl++;
					if (Mtrl > 0) {
						this.decoded.Indices.push([]);
					}
				}
			}
		}
		this.toMesh = function() {
			return new self.Mesh(this.decoded);
		}
		this.toObject = function(Mats) {
			if (Mats instanceof Array) {
				return self.CreateMeshObject(this.toMesh(),Mats);
			} else {
				return self.CreateMeshObject(this.toMesh(),new Array(this.decoded.Indices.length).fill(Mats || self.DefaultMaterial));
			}
		}
	}
	this.Japaneasetuples = [['ｳﾞ','ヴ'],['ｶﾞ','ガ'],['ｷﾞ','ギ'],['ｸﾞ','グ'],['ｹﾞ','ゲ'],['ｺﾞ','ゴ'],['ｻﾞ','ザ'],['ｼﾞ','ジ'],['ｽﾞ','ズ'],['ｾﾞ','ゼ'],['ｿﾞ','ゾ'],['ﾀﾞ','ダ'],['ﾁﾞ','ヂ'],['ﾂﾞ','ヅ'],['ﾃﾞ','デ'],['ﾄﾞ','ド'],['ﾊﾞ','バ'],['ﾊﾟ','パ'],['ﾋﾞ','ビ'],['ﾋﾟ','ピ'],['ﾌﾞ','ブ'],['ﾌﾟ','プ'],['ﾍﾞ','ベ'],['ﾍﾟ','ペ'],['ﾎﾞ','ボ'],['ﾎﾟ','ポ'],['｡','。'],['｢','「'],['｣','」'],['､','、'],['･','・'],['ｦ','ヲ'],['ｧ','ァ'],['ｨ','ィ'],['ｩ','ゥ'],['ｪ','ェ'],['ｫ','ォ'],['ｬ','ャ'],['ｭ','ュ'],['ｮ','ョ'],['ｯ','ッ'],['ｰ','ー'],['ｱ','ア'],['ｲ','イ'],['ｳ','ウ'],['ｴ','エ'],['ｵ','オ'],['ｶ','カ'],['ｷ','キ'],['ｸ','ク'],['ｹ','ケ'],['ｺ','コ'],['ｻ','サ'],['ｼ','シ'],['ｽ','ス'],['ｾ','セ'],['ｿ','ソ'],['ﾀ','タ'],['ﾁ','チ'],['ﾂ','ツ'],['ﾃ','テ'],['ﾄ','ト'],['ﾅ','ナ'],['ﾆ','ニ'],['ﾇ','ヌ'],['ﾈ','ネ'],['ﾉ','ノ'],['ﾊ','ハ'],['ﾋ','ヒ'],['ﾌ','フ'],['ﾍ','ヘ'],['ﾎ','ホ'],['ﾏ','マ'],['ﾐ','ミ'],['ﾑ','ム'],['ﾒ','メ'],['ﾓ','モ'],['ﾔ','ヤ'],['ﾕ','ユ'],['ﾖ','ヨ'],['ﾗ','ラ'],['ﾘ','リ'],['ﾙ','ル'],['ﾚ','レ'],['ﾛ','ロ'],['ﾜ','ワ'],['ﾝ','ン']];
	this.Japaneasetranslate = [['全ての親','ParentNode'],['操作中心','ControlNode'],['センター','Center'],['ｾﾝﾀｰ','Center'],['グループ','Group'],['グルーブ','Groove'],['キャンセル','Cancel'],['上半身','UpperBody'],['下半身','LowerBody'],['手首','Wrist'],['足首','Ankle'],['首','Neck'],['頭','Head'],['顔','Face'],['下顎','Chin'],['下あご','Chin'],['あご','Jaw'],['顎','Jaw'],['両目','Eyes'],['目','Eye'],['眉','Eyebrow'],['舌','Tongue'],['涙','Tears'],['泣き','Cry'],['歯','Teeth'],['照れ','Blush'],['青ざめ','Pale'],['ガーン','Gloom'],['汗','Sweat'],['怒','Anger'],['感情','Emotion'],['符','Marks'],['暗い','Dark'],['腰','Waist'],['髪','Hair'],['三つ編み','Braid'],['胸','Breast'],['乳','Boob'],['おっぱい','Tits'],['筋','Muscle'],['腹','Belly'],['鎖骨','Clavicle'],['肩','Shoulder'],['腕','Arm'],['うで','Arm'],['ひじ','Elbow'],['肘','Elbow'],['手','Hand'],['親指','Thumb'],['人指','IndexFinger'],['人差指','IndexFinger'],['中指','MiddleFinger'],['薬指','RingFinger'],['小指','LittleFinger'],['足','Leg'],['ひざ','Knee'],['つま','Toe'],['袖','Sleeve'],['新規','New'],['ボーン','Bone'],['捩','Twist'],['回転','Rotation'],['軸','Axis'],['ﾈｸﾀｲ','Necktie'],['ネクタイ','Necktie'],['ヘッドセット','Headset'],['飾り','Accessory'],['リボン','Ribbon'],['襟','Collar'],['紐','String'],['コード','Cord'],['イヤリング','Earring'],['メガネ','Eyeglasses'],['眼鏡','Glasses'],['帽子','Hat'],['ｽｶｰﾄ','Skirt'],['スカート','Skirt'],['パンツ','Pantsu'],['シャツ','Shirt'],['フリル','Frill'],['マフラー','Muffler'],['ﾏﾌﾗｰ','Muffler'],['服','Clothes'],['ブーツ','Boots'],['ねこみみ','CatEars'],['ジップ','Zip'],['ｼﾞｯﾌﾟ','Zip'],['ダミー','Dummy'],['ﾀﾞﾐｰ','Dummy'],['基','Category'],['あほ毛','Antenna'],['アホ毛','Antenna'],['モミアゲ','Sideburn'],['もみあげ','Sideburn'],['ツインテ','Twintail'],['おさげ','Pigtail'],['ひらひら','Flutter'],['調整','Adjustment'],['補助','Aux'],['右','Right'],['左','Left'],['前','Front'],['後ろ','Behind'],['後','Back'],['横','Side'],['中','Middle'],['上','Upper'],['下','Lower'],['親','Parent'],['先','Tip'],['パーツ','Part'],['光','Light'],['戻','Return'],['羽','Wing'],['根','Base'],['毛','Strand'],['尾','Tail'],['尻','Butt'],['０','0'],['１','1'],['２','2'],['３','3'],['４','4'],['５','5'],['６','6'],['７','7'],['８','8'],['９','9'],['ａ','a'],['ｂ','b'],['ｃ','c'],['ｄ','d'],['ｅ','e'],['ｆ','f'],['ｇ','g'],['ｈ','h'],['ｉ','i'],['ｊ','j'],['ｋ','k'],['ｌ','l'],['ｍ','m'],['ｎ','n'],['ｏ','o'],['ｐ','p'],['ｑ','q'],['ｒ','r'],['ｓ','s'],['ｔ','t'],['ｕ','u'],['ｖ','v'],['ｗ','w'],['ｘ','x'],['ｙ','y'],['ｚ','z'],['Ａ','A'],['Ｂ','B'],['Ｃ','C'],['Ｄ','D'],['Ｅ','E'],['Ｆ','F'],['Ｇ','G'],['Ｈ','H'],['Ｉ','I'],['Ｊ','J'],['Ｋ','K'],['Ｌ','L'],['Ｍ','M'],['Ｎ','N'],['Ｏ','O'],['Ｐ','P'],['Ｑ','Q'],['Ｒ','R'],['Ｓ','S'],['Ｔ','T'],['Ｕ','U'],['Ｖ','V'],['Ｗ','W'],['Ｘ','X'],['Ｙ','Y'],['Ｚ','Z'],['＋','+'],['－','-'],['＿','_'],['／','/'],['.','_']];
	this.Translate = function(u8arr) {
		var resu = Encoding.convert(u8arr, {from:'SJIS',type:'string'}).replaceAll("\x00","");
		for (var i=0; i<self.Japaneasetuples.length; i++) {
			resu = resu.replaceAll(self.Japaneasetuples[i][0],self.Japaneasetuples[i][1]);
		}
		for (var i=0; i<self.Japaneasetranslate.length; i++) {
			resu = resu.replaceAll(self.Japaneasetranslate[i][0],self.Japaneasetranslate[i][1]);
		}
		return resu;
	}
	this.VMDdecoder = function() {
		this.parsed = {};
		this.parse = function(data) {
			var idx = 0;
			var isNew = null;
			var KeyFrameCount = 1;
			var CurrentFrame = 0;
			var SpareData = null;
			var Part = 0;
			var result = [];
			var Bones = [];
			var BlendShapes = [];
			var NextKeyFrames = [];
			var header = Array.from(data.slice(0,30)).map(function(x){return String.fromCharCode(x)}).join("");
            isNew = header.includes('Vocaloid Motion Data 0002');
            if (isNew) {
                console.log("Vmd File Header:\n"+header+"\nIt is the new version of Vmd file.");
                data = data.slice(50);
            } else {
                console.log("Vmd File Header:\n"+header+"\nIt is the old version of Vmd file.");
                data = data.slice(40);
            }
			var d = data;
			KeyFrameCount = new Uint32Array(d.slice(0,4).buffer)[0];
			// KeyFrameCount = 100;
			console.log("KeyFrameCount: "+KeyFrameCount);
			d = d.slice(4);
			while (d.length >= 111 && Part == 0) {
				//utf16le
				var name = self.Translate(d.slice(0,15));
				var Frame = new Uint32Array(d.slice(15,19).buffer)[0];
				var P = new Float32Array(d.slice(19,31).buffer);
				var Q = new Float32Array(d.slice(31,47).buffer);
				if (!Bones.includes(name)) {
					Bones.push(name);
				}
				if (!result[Frame]) {
					result[Frame] = {Bones:{},BlendShapes:{},Camera:{}};
				}
				result[Frame].Bones[name] = {Position:P,Rotation:Q};
				CurrentFrame++;
				d = d.slice(111,Infinity);
				if (CurrentFrame >= KeyFrameCount) {
					CurrentFrame = 0;
					KeyFrameCount = new Uint32Array(d.slice(0,4).buffer)[0];
					console.log("KeyFrameCount: "+KeyFrameCount);
					d = d.slice(4,Infinity);
					Part = 1;
					break;
				}
			}
			while (d.length >= 23 && Part == 1) {
				var name = self.Translate(d.slice(0,15));
				var Frame = new Uint32Array(d.slice(15,19).buffer)[0];
				var Value = new Float32Array(d.slice(19,23).buffer)[0];
				if (!BlendShapes.includes(name)) {
					BlendShapes.push(name);
				}
				if (!result[Frame]) {
					result[Frame] = {Bones:{},BlendShapes:{},Camera:{}};
				}
				result[Frame].BlendShapes[name] = Value;
				CurrentFrame++;
				d = d.slice(23);
				if (CurrentFrame >= KeyFrameCount) {
					CurrentFrame = 0;
					KeyFrameCount = new Uint32Array(d.slice(0,4).buffer)[0];
					d = d.slice(4);
					Part = 2;
				}
			}
			while (d.length >= 61 && Part == 2) {
				var Frame = new Uint32Array(d.slice(0,4).buffer)[0];
				var D = new Float32Array(d.slice(4,8).buffer)[0];
				var P = new Float32Array(d.slice(8,20).buffer);
				var R = new Float32Array(d.slice(20,32).buffer);
				var Fov = new Uint32Array(d.slice(56,60).buffer)[0];
				if (!result[Frame]) {
					result[Frame] = {Bones:{},BlendShapes:{},Camera:{}};
				}
				result[Frame].Camera = {Position:P,Rotation:R,Distance:D,Fov:Fov};
				//console.log(P,Q);
				CurrentFrame++;
				d = d.slice(61);
				if (CurrentFrame >= KeyFrameCount) {
					console.log("Done");
					Part = 3;
				}
			}
			this.parsed = result;
		}
	}
	this.AssetFileImporter = function() {
		this.Assets = [];
		this.ImportFile = function(file,returnFunction) {
			if (file.name.endsWith(".fbx")) {
				var reader = new FileReader();
				reader.onload = function(e) {
					var data = new Uint8Array(e.target.result);
					
				}
				reader.readAsArrayBuffer(file);
			} else if (file.name.endsWith(".obj")) {
				var reader = new FileReader();
				reader.onload = function(e) {
					var data = new Uint8Array(e.target.result);
					var decoder = new Engine.OBJdecoder();
					decoder.parse(data);
					decoder.decodeData();
				}
				reader.readAsArrayBuffer(file);
			} else if (file.name.endsWith(".obj")) {
				var reader = new FileReader();
				reader.onload = function(e) {
					var data = new Uint8Array(e.target.result);
					var blob = new Blob([data]);
					var Img = new Image();
					Img.src = URL.createObjectURL(blob);
					Img.onload = function() {
						returnFunction(new Engine.Texture(Img));
					}
				}
				reader.readAsArrayBuffer(file);
			}
		}
	}
    this.DefaultVertexShader = new self.VertexShader(`#version 300 es
    in highp ivec3 aVertexDataIndexs;
    uniform highp sampler2D uVertexPoitions;
	uniform highp sampler2D uVertexNormals;
	uniform highp sampler2D uVertexUVs;
	
	uniform highp mat3x4 uViewMatrix;
	uniform highp vec3 uCamreaPosition;
    uniform highp vec4 uOptions;
	
	out highp vec3 WorldPosition;
	out highp vec3 PositionToCamera;
	out highp vec3 CameraSpacePosition;
	out highp vec3 Normal;
	out highp float depth;
	out highp vec2 UV;
	
    highp vec3 getVertexPosition(highp int idx) {
		highp int w = textureSize(uVertexPoitions,0)[0];
		return texelFetch(uVertexPoitions,ivec2(idx%w,idx/w),0).xyz;
	}
	highp vec3 getVertexNormal(highp int idx) {
		highp int w = textureSize(uVertexNormals,0)[0];
		return texelFetch(uVertexNormals,ivec2(idx%w,idx/w),0).xyz;
	}
	highp vec2 getVertexUV(highp int idx) {
		highp int w = textureSize(uVertexUVs,0)[0];
		return texelFetch(uVertexUVs,ivec2(idx%w,idx/w),0).xy;
	}
    highp vec3 WorldVectorToCamera(highp vec3 WorldVector) {
        return vec4(WorldVector,0.0)*uViewMatrix;
    }
    highp vec3 WorldPointToCamera(highp vec3 WorldPoint) {
        return vec4(WorldPoint,1.0)*uViewMatrix;
    }
    void main(void) {
		highp vec3 point = getVertexPosition(aVertexDataIndexs[0]);
        WorldPosition = point;
		Normal = getVertexNormal(aVertexDataIndexs[1]);
        point = WorldPointToCamera(point);
		UV = getVertexUV(aVertexDataIndexs[2]);
		CameraSpacePosition = point;
		PositionToCamera = uCamreaPosition-WorldPosition;
        point.z *= uOptions.z;
        gl_Position = vec4(point.xy*uOptions.xy,point.z-uOptions.w,point.z);
    }
    `);
    this.DebugShader = new this.Shader(self.DefaultVertexShader,new self.FragmentShader(`#version 300 es
        layout(location = 0) out highp vec4 color;
		layout(location = 1) out highp vec4 position;
		in highp vec3 PositionToCamera;
		in highp vec3 WorldPosition;
		in highp vec3 Normal;
        void main(void) {
			//position = vec4(WorldPosition,CameraSpacePosition.z);
			//color = vec4(mod(WorldPosition,vec3(1)),1);
			//color = vec4(abs(Normal),1);
			color = vec4((Normal+1.0)*0.5,1);
			//color = vec4((normalize(PositionToCamera)+1.0)*0.5,1);
        }
    `),[]);
    this.DefaultShader = new this.Shader(self.DefaultVertexShader,new self.FragmentShader(`#version 300 es
		highp vec3[24] fibSphere = vec3[24](vec3(0.08330922277203934,1.9982641400478363,0), vec3(-0.18386253656195853,1.984395334458658,-0.16843314468272927), vec3(0.03616444827285966,1.9567539787075074,0.4120752352343671), vec3(0.3499118507917456,1.9155319155950687,-0.45639804667875356), vec3(-0.721346997383237,1.8610152438246286,0.12759612697314385), vec3(0.7466459289412188,1.7935823323550246,0.4749550229857708), vec3(-0.26768567859714737,1.7137011943586609,-0.9957773817112359), vec3(-0.5393508864777872,1.6219262390104359,1.038487311653391), vec3(1.222186501017631,1.5188944236515596,-0.4463405499509058), vec3(-1.3153906106679338,1.40532083303388,-0.542974122410637), vec3(0.6506404788955875,1.2819937163266504,1.3903809113050654), vec3(0.48976880487264574,1.1497690163310286,-1.5614601265670538), vec3(-1.493781494289108,1.0095644288718937,0.8656769092888913), vec3(1.7624458009727377,0.8623530335973324,0.38746876529475605), vec3(-1.0755223265031053,0.7091565403907663,-1.5298198346257468), vec3(-0.2470735207029002,0.5510381982685012,1.906649306877507), vec3(1.5000777814051125,0.38909541597797437,-1.2642671422605438), vec3(-1.9856682975967084,0.22445214551289128,-0.08211361820094092), vec3(1.417057403081286,0.05825108140691417,1.410161383632188), vec3(-0.09224721287263171,-0.10835427005387264,-1.9949310273486458), vec3(-1.2693174548472759,-0.2742075966566752,1.5210665313377336), vec3(1.93398701596605,-0.438157805562027,-0.26021521765608185), vec3(-1.5663385360298747,-0.5990670123791483,-1.0898175559362722), vec3(0.4064104272907257,-0.7558184385690411,1.8065350958418789));
        layout(location = 0) out highp vec4 color;
		layout(location = 1) out highp vec4 position;
        uniform highp vec3 uColor;
		uniform sampler2D uTexture;
		uniform samplerCube uReflectionCubemap[4];
		uniform highp vec4 uReflectionPosition[4];
		uniform highp vec3 LightColors[24];
		uniform highp vec4 LightPositions[24];
		uniform highp float Reflectivity;
		uniform highp float Roughness;
		uniform highp float IOR;
		in highp vec3 PositionToCamera;
		in highp vec3 CameraSpacePosition;
		in highp vec3 WorldPosition;
		in highp vec3 Normal;
		in highp vec2 UV;
		const highp float Infinity = 1.0/0.0;
		highp ivec4 ReflectionResolutions;
		highp vec3 fastNormalize(highp vec3 vect) {
			vect = mix(vect,vect/dot(vect,vect),0.8);
			vect = mix(vect,vect/dot(vect,vect),0.7);
			vect = mix(vect,vect/dot(vect,vect),0.5);
			return vect;
		}
		highp vec3 RayMarch(highp vec3 pos, highp vec3 dir, highp float rough) {
			highp vec3 p = pos;
			highp vec3 v;
			bool hit;
			highp float dist = Infinity;
			highp float dtmp = Infinity;
			for (highp float i=0.025; i<1.0; i+=0.025) {
				p = pos-(dir*i/(1.0-i));
				v = abs(uReflectionPosition[0].xyz-p);
				hit = texture(uReflectionCubemap[0],uReflectionPosition[0].xyz-p).w-max(v.x,max(v.y,v.z)) < -0.08;
				if (hit) {
					v = abs(uReflectionPosition[1].xyz-p);
					hit = texture(uReflectionCubemap[1],uReflectionPosition[1].xyz-p).w-max(v.x,max(v.y,v.z)) < -0.08;
					if (hit) {
						v = abs(uReflectionPosition[2].xyz-p);
						hit = texture(uReflectionCubemap[2],uReflectionPosition[2].xyz-p).w-max(v.x,max(v.y,v.z)) < -0.08;
						if (hit) {
							v = abs(uReflectionPosition[3].xyz-p);
							hit = texture(uReflectionCubemap[3],uReflectionPosition[3].xyz-p).w-max(v.x,max(v.y,v.z)) < -0.08;
							if (hit) {
								dist = i/(1.0-i);
								break;
							}
						}
					}
				}
			}
			if (hit) {
				for (lowp int i=0; i<8; i++) {
					v = abs(uReflectionPosition[0].xyz-p);
					dtmp = texture(uReflectionCubemap[0],uReflectionPosition[0].xyz-p).w-max(v.x,max(v.y,v.z));
					v = abs(uReflectionPosition[1].xyz-p);
					dtmp = max(texture(uReflectionCubemap[1],uReflectionPosition[1].xyz-p).w-max(v.x,max(v.y,v.z)),dtmp);
					v = abs(uReflectionPosition[2].xyz-p);
					dtmp = max(texture(uReflectionCubemap[2],uReflectionPosition[2].xyz-p).w-max(v.x,max(v.y,v.z)),dtmp);
					v = abs(uReflectionPosition[3].xyz-p);
					dtmp = max(texture(uReflectionCubemap[3],uReflectionPosition[3].xyz-p).w-max(v.x,max(v.y,v.z)),dtmp);
					dist += min(dtmp,0.01);
					p = pos-(dir*dist);
				}
			}
			highp float count = 0.0;
			highp float s0 = texture(uReflectionCubemap[0],uReflectionPosition[0].xyz-p).w;
			highp float s1 = texture(uReflectionCubemap[1],uReflectionPosition[1].xyz-p).w;
			highp float s2 = texture(uReflectionCubemap[2],uReflectionPosition[2].xyz-p).w;
			highp float s3 = texture(uReflectionCubemap[3],uReflectionPosition[3].xyz-p).w;
			highp vec3 result;
			//rough = log2(rough*(dist*rough+1.0)+1.0);
			v = abs(uReflectionPosition[0].xyz-p);
			if (s0+0.5 >= max(v.x,max(v.y,v.z))) {
				result += textureLod(uReflectionCubemap[0],uReflectionPosition[0].xyz-p,rough*uReflectionPosition[0].w).rgb;
				count++;
			}
			v = abs(uReflectionPosition[1].xyz-p);
			if (s1+0.5 >= max(v.x,max(v.y,v.z))) {
				result += textureLod(uReflectionCubemap[1],uReflectionPosition[1].xyz-p,rough*uReflectionPosition[1].w).rgb;
				count++;
			}
			v = abs(uReflectionPosition[2].xyz-p);
			if (s2+0.5 >= max(v.x,max(v.y,v.z))) {
				result += textureLod(uReflectionCubemap[2],uReflectionPosition[2].xyz-p,rough*uReflectionPosition[2].w).rgb;
				count++;
			}
			v = abs(uReflectionPosition[3].xyz-p);
			if (s3+0.5 >= max(v.x,max(v.y,v.z))) {
				result += textureLod(uReflectionCubemap[3],uReflectionPosition[3].xyz-p,rough*uReflectionPosition[3].w).rgb;
				count++;
			}
			if (count == 0.0) {
				return vec3(0);
			} else {
				return result/count;
			}
		}
		highp float FresnelReflectionCoefficient(highp vec3 dir, highp vec3 norml, highp float ior) {
			highp float cosi = max(dot(dir,norml),0.0);
			highp float cost = -dot(refract(dir,norml,1.0/ior),norml)*ior;
			return clamp((cost-cosi)/(cosi+cost),0.04,1.0);
		}
		highp float liniarSpecular(highp float val, highp float d) {
			return max(((val-1.0)/d)+1.0,0.0);
		}
		highp float exponentalSpecular(highp float val, highp float d) {
			highp float k = min(1.0/(d*d),1000000.0);
			return pow(max(val,0.0),k)*(1.0+k)*0.1;
		}
        void main(void) {
			ReflectionResolutions = ivec4(textureSize(uReflectionCubemap[0],0).x,textureSize(uReflectionCubemap[1],0).x,textureSize(uReflectionCubemap[2],0).x,textureSize(uReflectionCubemap[3],0).x);
			position = vec4(WorldPosition,CameraSpacePosition.z);
			highp vec3 normal = normalize(Normal);
			highp vec3 ViewDir = normalize(PositionToCamera);
			highp vec3 refl = reflect(ViewDir,normal);
			highp float fresnel = FresnelReflectionCoefficient(ViewDir,normal,IOR);
			highp vec3 ReflectedLight;
			highp vec3 DirectLight;
			highp float reflection = mix(fresnel,1.0,Reflectivity);
			highp vec3 position = WorldPosition+(normal*0.0625);
			for (lowp int i = 0; i<24; i++) {
				highp vec3 lig = (LightPositions[i].xyz-(LightPositions[i].w*WorldPosition));
				highp float div = (dot(lig,lig)+LightPositions[i].w);
				if (div == 0.0) {
					break;
				}
				highp vec3 lignorm = normalize(lig);
				DirectLight += LightColors[i]*max(dot(lignorm,normal)/div,0.0);
			}
			if (!all(equal(ReflectionResolutions,ivec4(0)))) {
				ReflectedLight += RayMarch(position,refl,Roughness);
			}
			color = vec4(mix(uColor*texture(uTexture,UV).rgb*DirectLight,ReflectedLight,reflection),1);
			// color = vec4(vec3(fresnel),1.0);
        }
    `),[{Name:'uColor',Type:'vec3'},{Name:'uTexture',Type:'sampler2D'},{Name:'Reflectivity',Type:'float'},{Name:'Roughness',Type:'float'},{Name:'IOR',Type:'float'}]);
	this.DiffuseShader = new this.Shader(self.DefaultVertexShader,new self.FragmentShader(`#version 300 es
        layout(location = 0) out highp vec4 color;
		layout(location = 1) out highp vec4 position;
        uniform highp vec3 uColor;
		uniform sampler2D uTexture;
		uniform highp vec3 LightColors[24];
		uniform highp vec4 LightPositions[24];
		in highp vec3 CameraSpacePosition;
		in highp vec3 WorldPosition;
		in highp vec3 Normal;
		in highp vec2 UV;
		highp vec3 fastNormalize(highp vec3 vect) {
			vect = mix(vect,vect/dot(vect,vect),0.8);
			vect = mix(vect,vect/dot(vect,vect),0.7);
			vect = mix(vect,vect/dot(vect,vect),0.5);
			return vect;
		}
        void main(void) {
			position = vec4(WorldPosition,CameraSpacePosition.z);
			highp vec3 normal = normalize(Normal);
			highp vec3 DirectLight;
			for (lowp int i = 0; i<24; i++) {
				highp vec3 lig = (LightPositions[i].xyz-(LightPositions[i].w*WorldPosition));
				highp float div = (dot(lig,lig)+LightPositions[i].w);
				if (div == 0.0) {
					break;
				}
				DirectLight += LightColors[i]*max(dot(normalize(lig),normal)/div,0.0);
			}
			color = vec4(uColor*texture(uTexture,UV).rgb*DirectLight,1);
        }
    `),[{Name:'uColor',Type:'vec3'},{Name:'uTexture',Type:'sampler2D'}]);
	this.ToonShader = new this.Shader(self.DefaultVertexShader,new self.FragmentShader(`#version 300 es
        layout(location = 0) out highp vec4 color;
		layout(location = 1) out highp vec4 position;
        uniform highp vec3 uColor;
		uniform sampler2D uTexture;
		uniform highp vec3 LightColors[8];
		uniform highp vec4 LightPositions[8];
		in highp vec3 CameraSpacePosition;
		in highp vec3 WorldPosition;
		in highp vec3 Normal;
		in highp vec2 UV;
		highp vec3 fastNormalize(highp vec3 vect) {
			vect = mix(vect,vect/dot(vect,vect),0.8);
			vect = mix(vect,vect/dot(vect,vect),0.7);
			vect = mix(vect,vect/dot(vect,vect),0.5);
			return vect;
		}
        void main(void) {
			position = vec4(WorldPosition,CameraSpacePosition.z);
			highp vec3 normal = normalize(Normal);
			highp vec3 DirectLight;
			for (lowp int i = 0; i<24; i++) {
				highp vec3 lig = (LightPositions[i].xyz-(LightPositions[i].w*WorldPosition));
				highp float div = (dot(lig,lig)+LightPositions[i].w);
				if (div == 0.0) {
					break;
				}
				DirectLight += LightColors[i]*max(sign(dot(lig,normal)),0.0)/div;
			}
			color = vec4(uColor*texture(uTexture,UV).rgb*DirectLight,1);
        }
    `),[{Name:'uColor',Type:'vec3'},{Name:'uTexture',Type:'sampler2D'}]);
	this.UnlitShader = new this.Shader(self.DefaultVertexShader,new self.FragmentShader(`#version 300 es
        layout(location = 0) out highp vec4 color;
		layout(location = 1) out highp vec4 position;
        uniform highp vec3 uColor;
		uniform sampler2D uTexture;
		in highp vec3 CameraSpacePosition;
		in highp vec3 PositionToCamera;
		in highp vec3 WorldPosition;
		in highp vec3 Normal;
		in highp vec2 UV;
		highp vec3 fastNormalize(highp vec3 vect) {
			highp vec3 a = abs(vect);
			vect /= max(max(a.x,a.y),a.z);
			vect = mix(vect,vect/dot(vect,vect),0.45);
			return vect;
		}
        void main(void) {
			position = vec4(WorldPosition,CameraSpacePosition.z);
			color = vec4(uColor*texture(uTexture,UV).rgb,1);
			// color = vec4(uColor*texture(uTexture,UV).rgb*dot(normalize(PositionToCamera),normalize(Normal)),1.0);
        }
    `),[{Name:'uColor',Type:'vec3'},{Name:'uTexture',Type:'sampler2D'}]);
	//,{Name:'uCubemap',Type:'samplerCube',Value:8}
	this.DefaultTexture = new this.Texture((function(){
		var cvs = document.createElement("canvas");
        cvs.width = 1;
        cvs.height = 1;
        var ct = cvs.getContext('2d');
        ct.fillStyle="#FFFFFF";
        ct.fillRect(0,0,1,1);
        return cvs;
	})());
	this.CubemapTest = null;
    this.DefaultMaterial = new this.Material(this.DefaultShader,{uColor:[1,1,1],uTexture:this.DefaultTexture,Reflectivity:0,Roughness:0,IOR:1.3});
	//this.DefaultMaterial = new this.Material(this.DiffuseShader,{uColor:[1,1,1],uTexture:this.DefaultTexture});
	this.DiffuseMaterial = new this.Material(this.DiffuseShader,{uColor:[1,1,1],uTexture:this.DefaultTexture});
	this.XaxisMaterial = new this.Material(this.UnlitShader,{uColor:[1,0,0],uTexture:this.DefaultTexture});
	this.YaxisMaterial = new this.Material(this.UnlitShader,{uColor:[0,1,0],uTexture:this.DefaultTexture});
	this.ZaxisMaterial = new this.Material(this.UnlitShader,{uColor:[0,0,1],uTexture:this.DefaultTexture});
    this.DebugMaterial = new this.Material(this.DebugShader,{});
	this.GoldenRatio = 1.6180339887498948482045868343;
	this.ThetaM = 3.8832220774509327;
	//mshDta.push({VertexPositions:VertexPositions,VertexNormals:VertexNormals,VertexDataIndexs:VertexDataIndexs,VertexUVs:newUVs,Indices:newIndcs,Weights:newWeights,WeightIndices:newWeightIndices});
    this.DefaultPlaneMesh = new this.Mesh({VertexPositions:[[1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1]],VertexNormals:[[0,1,0]],VertexUVs:[[1,1],[0,1],[1,0],[0,0]],VertexDataIndexs:[[0,0,0],[1,0,1],[2,0,2],[3,0,3]],Indices:[[[0,2,1],[3,1,2]]]});
	this.DefaultCubeMesh = new this.Mesh({VertexPositions:[[0.5,0.5,0.5],[-0.5,0.5,0.5],[0.5,-0.5,0.5],[-0.5,-0.5,0.5],[0.5,0.5,-0.5],[-0.5,0.5,-0.5],[0.5,-0.5,-0.5],[-0.5,-0.5,-0.5]],VertexNormals:[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],VertexDataIndexs:[[0,0,0],[2,0,0],[4,0,0],[6,0,0],[1,1,0],[3,1,0],[5,1,0],[7,1,0],[0,2,0],[1,2,0],[4,2,0],[5,2,0],[2,3,0],[3,3,0],[6,3,0],[7,3,0],[0,4,0],[1,4,0],[2,4,0],[3,4,0],[4,5,0],[5,5,0],[6,5,0],[7,5,0]],Indices:[[[0,1,2],[3,2,1],[4,6,5],[7,5,6],[8,10,9],[11,9,10],[12,13,14],[15,14,13],[16,17,18],[19,18,17],[20,22,21],[23,21,22]]]});
	this.DefaultFlipedCubeMesh = new this.Mesh({VertexPositions:[[0.5,0.5,0.5],[-0.5,0.5,0.5],[0.5,-0.5,0.5],[-0.5,-0.5,0.5],[0.5,0.5,-0.5],[-0.5,0.5,-0.5],[0.5,-0.5,-0.5],[-0.5,-0.5,-0.5]],VertexNormals:[[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]],VertexDataIndexs:[[0,0,0],[2,0,0],[4,0,0],[6,0,0],[1,1,0],[3,1,0],[5,1,0],[7,1,0],[0,2,0],[1,2,0],[4,2,0],[5,2,0],[2,3,0],[3,3,0],[6,3,0],[7,3,0],[0,4,0],[1,4,0],[2,4,0],[3,4,0],[4,5,0],[5,5,0],[6,5,0],[7,5,0]],Indices:[[[0,2,1],[3,1,2],[4,5,6],[7,6,5],[8,9,10],[11,10,9],[12,14,13],[15,13,14],[16,18,17],[19,17,18],[20,21,22],[23,22,21]]]});
	this.DefaultCubeRoomMesh = new this.Mesh({VertexPositions:[[0.5,0.5,0.5],[-0.5,0.5,0.5],[0.5,-0.5,0.5],[-0.5,-0.5,0.5],[0.5,0.5,-0.5],[-0.5,0.5,-0.5],[0.5,-0.5,-0.5],[-0.5,-0.5,-0.5]],VertexNormals:[[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]],VertexUVs:[[0,0],[1.5,0],[0,1.5],[1.5,1.5]],VertexDataIndexs:[[0,0,0],[2,0,0],[4,0,0],[6,0,0],[1,1,0],[3,1,0],[5,1,0],[7,1,0],[0,2,0],[1,2,0],[4,2,0],[5,2,0],[2,3,3],[3,3,2],[6,3,1],[7,3,0],[0,4,0],[1,4,0],[2,4,0],[3,4,0],[4,5,0],[5,5,0],[6,5,0],[7,5,0]],Indices:[[[0,2,1],[3,1,2],[4,5,6],[7,6,5],[8,9,10],[11,10,9],[16,18,17],[19,17,18],[20,21,22],[23,22,21]],[[12,14,13],[15,13,14]]]});
	this.DefaultUVSphereMesh = new this.Mesh((function(){
		var MeshData = {VertexPositions:[],VertexNormals:[],VertexUVs:[],VertexDataIndexs:[],Indices:[[]]};
		var s = 32;
		var r = 48;
		MeshData.VertexPositions.push([0,0.5,0]);
		MeshData.VertexNormals.push([0,1,0]);
		for (var x=0; x<s; x++) {
			MeshData.VertexDataIndexs.push([MeshData.VertexPositions.length,MeshData.VertexNormals.length,MeshData.VertexUVs.length]);
			MeshData.VertexUVs.push([x/s,1]);
		}
		for (var y=1; y<r; y++) {
			var theta0 = Math.PI*y/r;
			var m0 = Math.cos(theta0);
			var m1 = Math.sqrt(1-(m0*m0));
			for (var x=0; x<s; x++) {
				var theta1 = self.Tau*x/s;
				var xyz = [Math.cos(theta1)*m1,m0,Math.sin(theta1)*m1];
				MeshData.VertexPositions.push([xyz[0]/2,xyz[1]/2,xyz[2]/2]);
				MeshData.VertexNormals.push(xyz);
				var idx = MeshData.VertexDataIndexs.length;
				MeshData.VertexDataIndexs.push([MeshData.VertexPositions.length,MeshData.VertexNormals.length,MeshData.VertexUVs.length]);
				MeshData.Indices[0].push([Math.max(idx-s,0),idx+1,idx]);
				MeshData.VertexUVs.push([x/s,1-(y/r)]);
				if (y > 1) {
					MeshData.Indices[0].push([Math.max(idx-s,0),idx,Math.max((idx-s)-1,0)]);
				}
			}
		}
		MeshData.VertexPositions.push([0,-0.5,0]);
		MeshData.VertexNormals.push([0,-1,0]);
		var idx = MeshData.VertexDataIndexs.length;
		for (var x=0; x<s; x++) {
			MeshData.VertexDataIndexs.push([MeshData.VertexPositions.length-1,MeshData.VertexNormals.length-1,MeshData.VertexUVs.length]);
			MeshData.VertexUVs.push([x/s,0]);
			MeshData.Indices[0].push([MeshData.VertexDataIndexs.length-1,idx-(s+x+1),idx-(s+x)]);
		}
		return MeshData;
	})());
	this.DefaultSphereMesh = new this.Mesh((function(){
		var MeshData = {VertexPositions:[],VertexNormals:[],VertexDataIndexs:[],Indices:[[]]};
		var len = 24;
		for (var i=0; i<len; i++) {
			var theta = self.ThetaM*i;
			var h = Math.cos(2*(i+0.5)/len);
			var m = Math.sqrt(1-(h*h));
			MeshData.VertexPositions.push([Math.cos(theta)*m,h,Math.sin(theta)*m]);
			MeshData.VertexNormals.push(MeshData.VertexPositions[MeshData.VertexPositions.length-1])
			MeshData.VertexDataIndexs.push([i,i,0]);
		}
		
		return MeshData;
	})());
	this.fixRadianAngle = function(agl) {
		agl = agl%self.Tau;
		if (agl > Math.PI) {
			return agl-self.Tau;
		}
		if (agl < -Math.PI) {
			return self.Tau+agl;
		}
		return agl;
	}
	this.fixDegreeAngle = function(agl) {
		agl = agl%360;
		if (agl > 180) {
			return agl-360;
		}
		if (agl < -180) {
			return 360+agl;
		}
		return agl;
	}
	this.Angle = function(angle,isInRadians) {
		this.value = isInRadians ? self.fixRadianAngle(angle) : self.fixDegreeAngle(angle);
		this.isInRadians = isInRadians;
		this.radians = function() {
			return this.isInRadians ? this.value : (this.value*self.DtoR);
		}
		this.degrees = function() {
			return this.isInRadians ? (this.value*self.RtoD) : this.value;
		}
		this.toRadians = function() {
			if (!this.isInRadians) {
				this.isInRadians = true;
				this.value = this.value*self.DtoR;
			}
		}
		this.toDegrees = function() {
			if (this.isInRadians) {
				this.isInRadians = false;
				this.value = this.value*self.RtoD;
			}
		}
		this.Add = function(a) {
			if (this.isInRadians) {
				return new self.Angle(this.value+a.radians(),true);
			} else {
				return new self.Angle(this.value+a.degrees(),false);
			}
		}
		this.Subtract = function(a) {
			if (this.isInRadians) {
				return new self.Angle(this.value-a.radians(),true);
			} else {
				return new self.Angle(this.value-a.degrees(),false);
			}
		}
		this.Oposite = function() {
			return new self.Angle(-this.value,this.isInRadians);
		}
	}
	this.randomAngle = function() {
		return new self.Angle(Math.random()*self.Tau,true);
	}
	this.randomQuaternion = function() {
		var q = new self.Quaternion([1,1,1,1]);
		while (q.Length() >= 1) {
			q.value = [(Math.random()*2)-1,(Math.random()*2)-1,(Math.random()*2)-1,(Math.random()*2)-1];
		}
		return q;
	}
    this.Components = {};
	this.Components.PointLight = function(color) {
		this.Enabled = true;
		this.Color = color;
		this.__proto__.getData = function() {
			if (this.Enabled) {
				var v = this.Parrent.getGlobalTransform().traslation;
				return {Position:[v.value[0],v.value[1],v.value[2],1],Vector:v,Color:this.Color};
			}
		}
	}
	this.Components.DirectionalLight = function(color) {
		this.Enabled = true;
		this.Color = color;
		this.__proto__.getData = function() {
			if (this.Enabled) {
				var v = this.Parrent.getGlobalTransform().matrix.value[1];
				return {Position:[v[0],v[1],v[2],0],Vector:new self.Vector3(v),Color:this.Color};
			}
		}
	}
    this.Components.MeshRenderer = function(mesh,materials) {
        this.Enabled = true;
        this.Mesh = mesh;
        this.Materials = materials;
		this.LightColors = [];
		this.LightPositions = [];
		this.ReflectionProbe = new Array(4).fill({Cubemap:null,Distence:Infinity,Position:{value:[0,0,0]},PosLod:[0,0,0,0]});
		var VertexPositionRenderBuffer = self.gl.createFramebuffer();
		var VertexPositionRenderTexture = self.gl.createTexture();
		self.gl.bindTexture(self.gl.TEXTURE_2D, VertexPositionRenderTexture);
		self.gl.texImage2D(self.gl.TEXTURE_2D, 0, self.gl.RGBA32F, mesh.VertexPositions.width, mesh.VertexPositions.height, 0, self.gl.RGBA, self.gl.FLOAT, null);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.CLAMP_TO_EDGE);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.CLAMP_TO_EDGE);
		self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, VertexPositionRenderBuffer);
		self.gl.framebufferTexture2D(self.gl.FRAMEBUFFER, self.gl.COLOR_ATTACHMENT0, self.gl.TEXTURE_2D, VertexPositionRenderTexture,0);
		var VertexNormalRenderBuffer = self.gl.createFramebuffer();
		var VertexNormalRenderTexture = self.gl.createTexture();
		self.gl.bindTexture(self.gl.TEXTURE_2D, VertexNormalRenderTexture);
		self.gl.texImage2D(self.gl.TEXTURE_2D, 0, self.gl.RGBA32F, mesh.VertexNormalDirections.width, mesh.VertexNormalDirections.height, 0, self.gl.RGBA, self.gl.FLOAT, null);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.CLAMP_TO_EDGE);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.CLAMP_TO_EDGE);
		self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, VertexNormalRenderBuffer);
		self.gl.framebufferTexture2D(self.gl.FRAMEBUFFER, self.gl.COLOR_ATTACHMENT0, self.gl.TEXTURE_2D, VertexNormalRenderTexture,0);
		this.VertexPositionRenderBuffer = VertexPositionRenderBuffer;
		this.VertexPositionRenderTexture = VertexPositionRenderTexture;
		this.VertexNormalRenderBuffer = VertexNormalRenderBuffer;
		this.VertexNormalRenderTexture = VertexNormalRenderTexture;
		this.Update = function() {
			var LightColors = [];
			var LightPositions = [];
			var Lghts = new Array(24).fill({D:Infinity,Pos:[0,0,0,0],Color:[0,0,0]});
			var ModelTransform = this.Parrent.getGlobalTransform();
			var max = Infinity;
			for (var i=0; i<self.Lights.length; i++) {
				var res = {D:self.Lights[i].Vector.Subtract(ModelTransform.traslation).LengthSquared(),Pos:self.Lights[i].Position,Color:self.Lights[i].Color};
				if (res.Pos[3] === 1 && Lghts.length > 24) {
					for (var j=0; j<Lghts.length; j++) {
						if (Lghts[j].D > res.D) {
							Lghts.splice(j,0,res);
							break;
						}
					}
				} else {
					Lghts.unshift(res);
				}
			}
			Lghts.splice(24,Infinity);
			for (var i=0; i<Lghts.length; i++) {
				LightColors.push(Lghts[i].Color);
				LightPositions.push(Lghts[i].Pos);
			}
			this.LightColors = LightColors.flat();
			this.LightPositions = LightPositions.flat();
			
			var Refl = new Array(4).fill({Cubemap:null,Distence:Infinity,Position:{value:[0,0,0]},PosLod:[0,0,0,0]});
			for (var i=0; i<self.ReflectionProbes.length; i++) {
				var d = self.ReflectionProbes[i].Position.Subtract(ModelTransform.traslation).LengthSquared();
				if (d < Refl[3].Distence) {
					Refl.unshift(self.ReflectionProbes[i]);
					Refl.pop();
				}
			}
			this.ReflectionProbe = Refl;

			var ModelTransform = ModelTransform.ToMatrixTransposed3x4().value;
			var VPdims = self.gl.getParameter(self.gl.VIEWPORT);
			self.gl.viewport(0,0,65536,65536);
			self.gl.useProgram(self.MeshVertexMapingShader.program);
			self.gl.disable(self.gl.DEPTH_TEST);
			self.gl.uniform1i(self.MeshVertexMapingShader.notNormals,true);
			self.gl.activeTexture(self.gl.TEXTURE0);
			self.gl.bindTexture(self.gl.TEXTURE_2D,this.Mesh.VertexPositions.texture);
			self.gl.uniformMatrix3x4fv(self.MeshVertexMapingShader.Uniforms.uModelMatrix.Location,false,ModelTransform.flat());
			self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, VertexPositionRenderBuffer);
			self.gl.bindBuffer(self.gl.ELEMENT_ARRAY_BUFFER, self.VertexMapingIndexBuffer);
			self.gl.drawElements(self.gl.TRIANGLES,6,self.gl.UNSIGNED_SHORT,0);

			self.gl.uniform1i(self.MeshVertexMapingShader.notNormals,false);
			self.gl.activeTexture(self.gl.TEXTURE0);
			self.gl.bindTexture(self.gl.TEXTURE_2D,this.Mesh.VertexNormalDirections.texture);
			self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, VertexNormalRenderBuffer);
			self.gl.drawElements(self.gl.TRIANGLES,6,self.gl.UNSIGNED_SHORT,0);
			self.gl.enable(self.gl.DEPTH_TEST);

			self.gl.viewport(VPdims[0],VPdims[1],VPdims[2],VPdims[3]);
			self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, null);
		}
        this.render = function(CameraTransformInverse,CamreaViewPortComponent,CamPos) {
            if (this.Enabled) {
				self.gl.activeTexture(self.gl.TEXTURE0);
				self.gl.bindTexture(self.gl.TEXTURE_2D,VertexPositionRenderTexture);

				self.gl.activeTexture(self.gl.TEXTURE1);
				self.gl.bindTexture(self.gl.TEXTURE_2D,VertexNormalRenderTexture);

				self.gl.activeTexture(self.gl.TEXTURE2);
				self.gl.bindTexture(self.gl.TEXTURE_2D,this.Mesh.VertexUVPositions.texture);
				
				self.gl.activeTexture(self.gl.TEXTURE3);
				self.gl.bindTexture(self.gl.TEXTURE_CUBE_MAP, this.ReflectionProbe[0].Cubemap);
				
				self.gl.activeTexture(self.gl.TEXTURE4);
				self.gl.bindTexture(self.gl.TEXTURE_CUBE_MAP, this.ReflectionProbe[1].Cubemap);
				
				self.gl.activeTexture(self.gl.TEXTURE5);
				self.gl.bindTexture(self.gl.TEXTURE_CUBE_MAP, this.ReflectionProbe[2].Cubemap);
				
				self.gl.activeTexture(self.gl.TEXTURE6);
				self.gl.bindTexture(self.gl.TEXTURE_CUBE_MAP, this.ReflectionProbe[3].Cubemap);
				
				var VPdims = self.gl.getParameter(self.gl.VIEWPORT);
                for (var l=0; l<this.Mesh.IndexBuffers.length; l++) {
                    if (this.Materials[l]) {
                        self.gl.useProgram(this.Materials[l].shader.program);
						self.gl.uniform3fv(this.Materials[l].shader.CamreaPosition,CamPos);
						self.gl.uniform3fv(this.Materials[l].shader.LightColors,this.LightColors);
						self.gl.uniform4fv(this.Materials[l].shader.LightPositions,this.LightPositions);
						self.gl.uniform4fv(this.Materials[l].shader.ReflectionPosition,this.ReflectionProbe[0].PosLod.concat(this.ReflectionProbe[1].PosLod).concat(this.ReflectionProbe[2].PosLod).concat(this.ReflectionProbe[3].PosLod));
						//uReflectionPosition
                        self.gl.uniformMatrix3x4fv(this.Materials[l].shader.ViewMatrix,false,CameraTransformInverse.value.flat());
                        var F = Math.tan((Math.PI/360)*CamreaViewPortComponent.FeildOfView)*(Math.max(self.canvas.width,self.canvas.height)/Math.min(self.canvas.width,self.canvas.height));
                        var m = Math.max(VPdims[2],VPdims[3]);
                        self.gl.uniform4fv(this.Materials[l].shader.Options, [m/VPdims[2],m/VPdims[3],F,CamreaViewPortComponent.ClipDistance]);
                        var Ukeys = Object.keys(this.Materials[l].shader.Uniforms);
                        var texIdx = 7;
                        for (var i=0; i<Ukeys.length; i++) {
                            var U = this.Materials[l].shader.Uniforms[Ukeys[i]];
                            var value = this.Materials[l].Values[Ukeys[i]];
                            if (U && value || value === 0) {
                                if (U.Type == "float") {
                                    self.gl.uniform1f(U.Location,value);
                                } else if (U.Type == "vec3") {
                                    self.gl.uniform3fv(U.Location,value);
                                } else if (U.Type == "vec4") {
                                    self.gl.uniform4fv(U.Location,value);
                                } else if (U.Type == "mat3") {
                                    self.gl.uniformMatrix3fv(U.Location,false,value.flat());
                                } else if (U.Type == "mat4") {
                                    self.gl.uniformMatrix4fv(U.Location,false,value.flat());
                                } else if (U.Type == "sampler2D") {
                                    self.gl.activeTexture(self.gl.TEXTURE0+texIdx);
                                    self.gl.bindTexture(self.gl.TEXTURE_2D,value.texture);
                                    self.gl.uniform1i(U.Location,texIdx);
                                    texIdx++;
                                }
                            }
                        }
						self.gl.uniform1i(this.Materials[l].shader.VertexPoitions,0);
						
						self.gl.uniform1i(this.Materials[l].shader.VertexNormals,1);
						
						self.gl.uniform1i(this.Materials[l].shader.VertexUVs,2);
						
						self.gl.uniform1iv(this.Materials[l].shader.ReflectionCubemap,[3,4,5,6]);
						
                        self.gl.bindBuffer(self.gl.ARRAY_BUFFER, this.Mesh.VertexDataIndexs);
                        self.gl.vertexAttribIPointer(this.Materials[l].shader.VertexDataIndexs,3,self.gl.INT,0,0);
                        self.gl.bindBuffer(self.gl.ELEMENT_ARRAY_BUFFER, this.Mesh.IndexBuffers[l]);
                        self.gl.drawElements(self.gl.TRIANGLES,this.Mesh.l[l],self.gl.UNSIGNED_SHORT,0);
                    } else {
                        break;
                    }
                }
            }
        }
    }
	/*
	this.SkinnedMeshBlendShapeVertexMapingShader = new this.VertexMapingShader(new self.FragmentShader(`#version 300 es
		out highp vec4 VertexPosition;
		
		uniform highp sampler2D uVertexPositions;
		uniform highp isampler2D uBlendShapeIndexs;
		uniform highp sampler2D uBlendShapes;
		uniform highp float uBlendShapeWeight;
		
		void main(void) {
			highp ivec2 Pos = ivec2(gl_FragCoord.xy-0.5);
			highp vec3 Vertex = texelFetch(uVertexPositions,Pos,0).xyz;
			highp int idx = texelFetch(uBlendShapeIndexs,Pos,0)[0];
			if (idx >= 0) {
				highp int Wid = textureSize(uBlendShapes,0)[0];
				Vertex += texelFetch(uBlendShapes,ivec2(idx%Wid,idx/Wid),0).xyz*uBlendShapeWeight;
				VertexPosition = vec4(Vertex,1);
			} else {
				VertexPosition = vec4(Vertex,1);
			}
		}
	`),[{Name:'uVertexPositions',Type:'sampler2D',Value:0},{Name:'uBlendShapeIndexs',Type:'isampler2D',Value:1},{Name:'uBlendShapes',Type:'sampler2D',Value:2}]);
	*/
	this.SkinnedMeshRendererTransformTextures = [];
    this.Components.SkinnedMeshRenderer = function(SkinnedMesh,materials) {
        this.Enabled = true;
        this.SkinnedMesh = SkinnedMesh;
        this.InversePoseTransforms = [];
        this.DefaultPose = SkinnedMesh.DefaultPose;
        for (var i=0; i<SkinnedMesh.DefaultPose.length; i++) {
            this.InversePoseTransforms.push(SkinnedMesh.DefaultPose[i].InverseTransform());
        }
        this.Materials = materials;
        this.BoneRefrences = new Array(Object.keys(SkinnedMesh.Bones).length);
        this.SetBoneRefrence = function(BoneID,Obj) {
            this.BoneRefrences[BoneID] = Obj;
        }
		this.LightColors = [];
		this.LightPositions = [];
		this.ReflectionProbe = new Array(4).fill({Cubemap:null,Distence:Infinity,Position:{value:[0,0,0]},PosLod:[0,0,0,0]});
		var VertexPositionRenderBuffer = self.gl.createFramebuffer();
		var VertexPositionRenderTexture = self.gl.createTexture();
		self.gl.bindTexture(self.gl.TEXTURE_2D, VertexPositionRenderTexture);
		self.gl.texImage2D(self.gl.TEXTURE_2D, 0, self.gl.RGBA32F, SkinnedMesh.VertexPositions.width, SkinnedMesh.VertexPositions.height, 0, self.gl.RGBA, self.gl.FLOAT, null);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.CLAMP_TO_EDGE);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.CLAMP_TO_EDGE);
		self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, VertexPositionRenderBuffer);
		self.gl.framebufferTexture2D(self.gl.FRAMEBUFFER, self.gl.COLOR_ATTACHMENT0, self.gl.TEXTURE_2D, VertexPositionRenderTexture,0);
		var VertexBlendShapePositionRenderBuffer = self.gl.createFramebuffer();
		var VertexBlendShapePositionRenderTexture = self.gl.createTexture();
		self.gl.bindTexture(self.gl.TEXTURE_2D, VertexBlendShapePositionRenderTexture);
		self.gl.texImage2D(self.gl.TEXTURE_2D, 0, self.gl.RGBA32F, SkinnedMesh.VertexPositions.width, SkinnedMesh.VertexPositions.height, 0, self.gl.RGBA, self.gl.FLOAT, null);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.CLAMP_TO_EDGE);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.CLAMP_TO_EDGE);
		self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, VertexBlendShapePositionRenderBuffer);
		self.gl.framebufferTexture2D(self.gl.FRAMEBUFFER, self.gl.COLOR_ATTACHMENT0, self.gl.TEXTURE_2D, VertexBlendShapePositionRenderTexture,0);
		var VertexNormalRenderBuffer = self.gl.createFramebuffer();
		var VertexNormalRenderTexture = self.gl.createTexture();
		self.gl.bindTexture(self.gl.TEXTURE_2D, VertexNormalRenderTexture);
		self.gl.texImage2D(self.gl.TEXTURE_2D, 0, self.gl.RGBA32F, SkinnedMesh.VertexNormalDirections.width, SkinnedMesh.VertexNormalDirections.height, 0, self.gl.RGBA, self.gl.FLOAT, null);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.CLAMP_TO_EDGE);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.CLAMP_TO_EDGE);
		self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, VertexNormalRenderBuffer);
		self.gl.framebufferTexture2D(self.gl.FRAMEBUFFER, self.gl.COLOR_ATTACHMENT0, self.gl.TEXTURE_2D, VertexNormalRenderTexture,0);
		this.VertexPositionRenderBuffer = VertexPositionRenderBuffer;
		this.VertexPositionRenderTexture = VertexPositionRenderTexture;
		this.VertexNormalRenderBuffer = VertexNormalRenderBuffer;
		this.VertexNormalRenderTexture = VertexNormalRenderTexture;
		this.TransformTextureSize = self.calculateMostEfficientTextureSize(this.DefaultPose.length);
		this.BlendShapeValues = {};
		for (var i=0; i<SkinnedMesh.BlendShapes.length; i++) {
			this.BlendShapeValues[SkinnedMesh.BlendShapes[i].Name] = 0;
		}
		this.Update = function() {
			if (this.Enabled) {
				var LightColors = [];
				var LightPositions = [];
				var Lghts = [];
				var ModelTransform = this.Parrent.getGlobalTransform();
				var max = Infinity;
				for (var i=0; i<self.Lights.length; i++) {
					var res = {D:self.Lights[i].Vector.Subtract(ModelTransform.traslation).LengthSquared(),Pos:self.Lights[i].Position,Color:self.Lights[i].Color};
					if (res.Pos[3] === 1 && Lghts.length > 28) {
						for (var j=0; j<Lghts.length; j++) {
							if (Lghts[j].D > res.D) {
								Lghts.splice(j,0,res);
								break;
							}
						}
					} else {
						Lghts.unshift(res);
					}
				}
				Lghts.splice(24,Infinity);
				for (var i=0; i<Lghts.length; i++) {
					LightColors.push(Lghts[i].Color);
					LightPositions.push(Lghts[i].Pos);
				}
				this.LightColors = LightColors.flat();
				this.LightPositions = LightPositions.flat();
				
				var Refl = new Array(4).fill({Cubemap:null,Distence:Infinity,Position:{value:[0,0,0]},PosLod:[0,0,0,0]});
				for (var i=0; i<self.ReflectionProbes.length; i++) {
					var d = self.ReflectionProbes[i].Position.Subtract(ModelTransform.traslation).LengthSquared();
					if (d < Refl[3].Distence) {
						Refl.unshift(self.ReflectionProbes[i]);
						Refl.pop();
					}
				}
				this.ReflectionProbe = Refl;
				
				var ModelTransform = ModelTransform.ToMatrixTransposed3x4().value;
				var TextureData = [new Float32Array(this.DefaultPose.length*4),new Float32Array(this.DefaultPose.length*4),new Float32Array(this.DefaultPose.length*4)];
                var ModelTransforms = [];
                for (var i=0; i<this.BoneRefrences.length; i++) {
                    ModelTransforms.push(this.BoneRefrences[i].getGlobalTransform().TransformTransform(this.InversePoseTransforms[i]).ToMatrixTransposed3x4().value);
					TextureData[0].set(ModelTransforms[i][0],i*4);
					TextureData[1].set(ModelTransforms[i][1],i*4);
					TextureData[2].set(ModelTransforms[i][2],i*4);
                }
				if (!self.SkinnedMeshRendererTransformTextures[0]) {
					self.SkinnedMeshRendererTransformTextures[0] = self.gl.createTexture();
					self.gl.bindTexture(self.gl.TEXTURE_2D, self.SkinnedMeshRendererTransformTextures[0]);
					self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
					self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.NEAREST);
					self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.CLAMP_TO_EDGE);
					self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.CLAMP_TO_EDGE);
				}
				self.gl.bindTexture(self.gl.TEXTURE_2D, self.SkinnedMeshRendererTransformTextures[0]);
				self.gl.texImage2D(self.gl.TEXTURE_2D, 0, self.gl.RGBA32F, this.TransformTextureSize[0], this.TransformTextureSize[1], 0, self.gl.RGBA, self.gl.FLOAT, TextureData[0]);
				
				if (!self.SkinnedMeshRendererTransformTextures[1]) {
					self.SkinnedMeshRendererTransformTextures[1] = self.gl.createTexture();
					self.gl.bindTexture(self.gl.TEXTURE_2D, self.SkinnedMeshRendererTransformTextures[1]);
					self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
					self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.NEAREST);
					self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.CLAMP_TO_EDGE);
					self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.CLAMP_TO_EDGE);
				}
				self.gl.bindTexture(self.gl.TEXTURE_2D, self.SkinnedMeshRendererTransformTextures[1]);
				self.gl.texImage2D(self.gl.TEXTURE_2D, 0, self.gl.RGBA32F, this.TransformTextureSize[0], this.TransformTextureSize[1], 0, self.gl.RGBA, self.gl.FLOAT, TextureData[1]);
				
				if (!self.SkinnedMeshRendererTransformTextures[2]) {
					self.SkinnedMeshRendererTransformTextures[2] = self.gl.createTexture();
					self.gl.bindTexture(self.gl.TEXTURE_2D, self.SkinnedMeshRendererTransformTextures[2]);
					self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
					self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.NEAREST);
					self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.CLAMP_TO_EDGE);
					self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.CLAMP_TO_EDGE);
				}
				self.gl.bindTexture(self.gl.TEXTURE_2D, self.SkinnedMeshRendererTransformTextures[2]);
				self.gl.texImage2D(self.gl.TEXTURE_2D, 0, self.gl.RGBA32F, this.TransformTextureSize[0], this.TransformTextureSize[1], 0, self.gl.RGBA, self.gl.FLOAT, TextureData[2]);
				//console.log(TextureData);
				
				var VPdims = self.gl.getParameter(self.gl.VIEWPORT);
				self.gl.viewport(0,0,65536,65536);
				self.gl.disable(self.gl.DEPTH_TEST);

				self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, VertexBlendShapePositionRenderBuffer);
				self.gl.clearColor(0.0,0.0,0.0, 1.0);
				self.gl.clear(self.gl.COLOR_BUFFER_BIT);
				self.gl.useProgram(self.SkinnedMeshBlendShapeVertexMapingShader.program);
				self.gl.enable(self.gl.BLEND);
				self.gl.blendFunc(self.gl.SRC_ALPHA, self.gl.ONE);
				for (var i=0; i<this.SkinnedMesh.BlendShapes.length; i++) {
					var BlendShapeName = this.SkinnedMesh.BlendShapes[i].Name;
					var BlendShapeValue = this.BlendShapeValues[BlendShapeName];
					if (BlendShapeValue > 0) {
						self.gl.activeTexture(self.gl.TEXTURE0);
						self.gl.bindTexture(self.gl.TEXTURE_2D, this.SkinnedMesh.BlendShapes[i].Indexs.texture);
						self.gl.activeTexture(self.gl.TEXTURE1);
						self.gl.bindTexture(self.gl.TEXTURE_2D, this.SkinnedMesh.BlendShapes[i].BlendShapeNormals.texture);
						self.gl.uniform1f(self.SkinnedMeshBlendShapeVertexMapingShader.BlendShapeWeight,BlendShapeValue);
						self.gl.drawElements(self.gl.TRIANGLES,6,self.gl.UNSIGNED_SHORT,0);
					}
				}
				self.gl.disable(self.gl.BLEND);

				self.gl.useProgram(self.SkinnedMeshVertexMapingShader.program);
				self.gl.uniform1i(self.SkinnedMeshVertexMapingShader.notNormals,true);
				self.gl.activeTexture(self.gl.TEXTURE0);
				self.gl.bindTexture(self.gl.TEXTURE_2D,this.SkinnedMesh.VertexPositions.texture);
				self.gl.activeTexture(self.gl.TEXTURE1);
				self.gl.bindTexture(self.gl.TEXTURE_2D, self.SkinnedMeshRendererTransformTextures[0]);
				self.gl.activeTexture(self.gl.TEXTURE2);
				self.gl.bindTexture(self.gl.TEXTURE_2D, self.SkinnedMeshRendererTransformTextures[1]);
				self.gl.activeTexture(self.gl.TEXTURE3);
				self.gl.bindTexture(self.gl.TEXTURE_2D, self.SkinnedMeshRendererTransformTextures[2]);
				self.gl.activeTexture(self.gl.TEXTURE4);
				self.gl.bindTexture(self.gl.TEXTURE_2D,this.SkinnedMesh.WeightIndexs.texture);
				self.gl.activeTexture(self.gl.TEXTURE5);
				self.gl.bindTexture(self.gl.TEXTURE_2D,this.SkinnedMesh.Weights.texture);
				self.gl.activeTexture(self.gl.TEXTURE6);
				self.gl.bindTexture(self.gl.TEXTURE_2D,VertexBlendShapePositionRenderTexture);
				self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, VertexPositionRenderBuffer);
				self.gl.bindBuffer(self.gl.ELEMENT_ARRAY_BUFFER, self.VertexMapingIndexBuffer);
				self.gl.drawElements(self.gl.TRIANGLES,6,self.gl.UNSIGNED_SHORT,0);

				//NormalWeights NormalWeightIndexs
				self.gl.uniform1i(self.SkinnedMeshVertexMapingShader.notNormals,false);
				self.gl.activeTexture(self.gl.TEXTURE0);
				self.gl.bindTexture(self.gl.TEXTURE_2D,this.SkinnedMesh.VertexNormalDirections.texture);
				self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, VertexNormalRenderBuffer);
				self.gl.drawElements(self.gl.TRIANGLES,6,self.gl.UNSIGNED_SHORT,0);
				self.gl.enable(self.gl.DEPTH_TEST);
				
				self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, null);
				self.gl.viewport(VPdims[0],VPdims[1],VPdims[2],VPdims[3]);
			}
		}
        this.render = function(CameraTransformInverse,CamreaViewPortComponent,CamPos) {
            if (this.Enabled) {
				var VPdims = self.gl.getParameter(self.gl.VIEWPORT);
				
				self.gl.activeTexture(self.gl.TEXTURE0);
				self.gl.bindTexture(self.gl.TEXTURE_2D,VertexPositionRenderTexture);

				self.gl.activeTexture(self.gl.TEXTURE1);
				self.gl.bindTexture(self.gl.TEXTURE_2D,VertexNormalRenderTexture);

				self.gl.activeTexture(self.gl.TEXTURE2);
				self.gl.bindTexture(self.gl.TEXTURE_2D,this.SkinnedMesh.VertexUVPositions.texture);
				
				self.gl.activeTexture(self.gl.TEXTURE3);
				self.gl.bindTexture(self.gl.TEXTURE_CUBE_MAP, this.ReflectionProbe[0].Cubemap);
				
				self.gl.activeTexture(self.gl.TEXTURE4);
				self.gl.bindTexture(self.gl.TEXTURE_CUBE_MAP, this.ReflectionProbe[1].Cubemap);
				
				self.gl.activeTexture(self.gl.TEXTURE5);
				self.gl.bindTexture(self.gl.TEXTURE_CUBE_MAP, this.ReflectionProbe[2].Cubemap);
				
				self.gl.activeTexture(self.gl.TEXTURE6);
				self.gl.bindTexture(self.gl.TEXTURE_CUBE_MAP, this.ReflectionProbe[3].Cubemap);
								
                for (var l=0; l<this.SkinnedMesh.IndexBuffers.length; l++) {
                    if (this.Materials[l]) {
                        self.gl.useProgram(this.Materials[l].shader.program);
						self.gl.uniform3fv(this.Materials[l].shader.CamreaPosition,CamPos);
						self.gl.uniform3fv(this.Materials[l].shader.LightColors,this.LightColors);
						self.gl.uniform4fv(this.Materials[l].shader.LightPositions,this.LightPositions);
						self.gl.uniform4fv(this.Materials[l].shader.ReflectionPosition,this.ReflectionProbe[0].PosLod.concat(this.ReflectionProbe[1].PosLod).concat(this.ReflectionProbe[2].PosLod).concat(this.ReflectionProbe[3].PosLod));
                        self.gl.uniformMatrix3x4fv(this.Materials[l].shader.ViewMatrix,false,CameraTransformInverse.value.flat());
                        var F = Math.tan((Math.PI/360)*CamreaViewPortComponent.FeildOfView)*(Math.max(self.canvas.width,self.canvas.height)/Math.min(self.canvas.width,self.canvas.height));
                        var m = Math.max(VPdims[2],VPdims[3]);
                        self.gl.uniform4fv(this.Materials[l].shader.Options, [m/VPdims[2],m/VPdims[3],F,CamreaViewPortComponent.ClipDistance]);
                        var Ukeys = Object.keys(this.Materials[l].shader.Uniforms);
                        var texIdx = 7;
                        for (var i=0; i<Ukeys.length; i++) {
                            var U = this.Materials[l].shader.Uniforms[Ukeys[i]];
                            var value = this.Materials[l].Values[Ukeys[i]];
                            if (U && value || value === 0) {
                                if (U.Type == "float") {
                                    self.gl.uniform1f(U.Location,value);
                                } else if (U.Type == "vec3") {
                                    self.gl.uniform3fv(U.Location,value);
                                } else if (U.Type == "vec4") {
                                    self.gl.uniform4fv(U.Location,value);
                                } else if (U.Type == "mat3") {
                                    self.gl.uniformMatrix3fv(U.Location,false,value.flat());
                                } else if (U.Type == "mat4") {
                                    self.gl.uniformMatrix4fv(U.Location,false,value.flat());
                                } else if (U.Type == "sampler2D") {
                                    self.gl.activeTexture(self.gl.TEXTURE0+texIdx);
                                    self.gl.bindTexture(self.gl.TEXTURE_2D,value.texture);
                                    self.gl.uniform1i(U.Location,texIdx);
                                    texIdx++;
                                }
                            }
                        }
						self.gl.uniform1i(this.Materials[l].shader.VertexPoitions,0);
						
						self.gl.uniform1i(this.Materials[l].shader.VertexNormals,1);
						
						self.gl.uniform1i(this.Materials[l].shader.VertexUVs,2);
						
						self.gl.uniform1iv(this.Materials[l].shader.ReflectionCubemap,[3,4,5,6]);
						
                        self.gl.bindBuffer(self.gl.ARRAY_BUFFER, this.SkinnedMesh.VertexDataIndexs);
                        self.gl.vertexAttribIPointer(this.Materials[l].shader.VertexDataIndexs,3,self.gl.INT,0,0);
                        self.gl.bindBuffer(self.gl.ELEMENT_ARRAY_BUFFER, this.SkinnedMesh.IndexBuffers[l]);
                        self.gl.drawElements(self.gl.TRIANGLES,this.SkinnedMesh.l[l],self.gl.UNSIGNED_SHORT,0);
                    } else {
                        break;
                    }
                }
            }
        }
    }
	this.RenderQue = [];
	this.Lights = [];
	this.ReflectionProbes = [];
    this.Components.CamreaViewPort = function(canvas,resolution,feildOfView,clipDistance,isVR,IPD,pp) {
        this.Enabled = true;
        this.ClipDistance = clipDistance || 0.2;
        this.FeildOfView = feildOfView || 100;
        this.isVR = isVR || false;
		this.PostProcessing = (pp === undefined) ? true : !!pp;
        this.IPD = IPD || 0.058;
		this.Resolution = resolution;
        // this.IPD = IPD || 1;
        this.canvas = canvas;
		this.PreviousMat = new self.Matrix3x3();
        var dis = this;
		this.FrameBuffer = self.gl.createFramebuffer();
		this.DepthBuffer = self.gl.createRenderbuffer();
		self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, this.FrameBuffer);
		self.gl.bindRenderbuffer(self.gl.RENDERBUFFER, this.DepthBuffer);
		self.gl.renderbufferStorage(self.gl.RENDERBUFFER, self.gl.DEPTH_COMPONENT24, this.Resolution[0], this.Resolution[1]);
		self.gl.framebufferRenderbuffer(self.gl.FRAMEBUFFER, self.gl.DEPTH_ATTACHMENT, self.gl.RENDERBUFFER, this.DepthBuffer);
		self.gl.drawBuffers([self.gl.COLOR_ATTACHMENT0,self.gl.COLOR_ATTACHMENT1]);
		this.ColorRenderTexture = self.gl.createTexture();
		self.gl.bindTexture(self.gl.TEXTURE_2D, this.ColorRenderTexture);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.CLAMP_TO_EDGE);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.CLAMP_TO_EDGE);
		self.gl.texImage2D(self.gl.TEXTURE_2D, 0, self.gl.RGBA32F, this.Resolution[0], this.Resolution[1], 0, self.gl.RGBA, self.gl.FLOAT, null);
		self.gl.framebufferTexture2D(self.gl.FRAMEBUFFER, self.gl.COLOR_ATTACHMENT0, self.gl.TEXTURE_2D, this.ColorRenderTexture, 0);
		this.PositionRenderTexture = self.gl.createTexture();
		self.gl.bindTexture(self.gl.TEXTURE_2D, this.PositionRenderTexture);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.CLAMP_TO_EDGE);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.CLAMP_TO_EDGE);
		self.gl.texImage2D(self.gl.TEXTURE_2D, 0, self.gl.RGBA32F, this.Resolution[0], this.Resolution[1], 0, self.gl.RGBA, self.gl.FLOAT, null);
		self.gl.framebufferTexture2D(self.gl.FRAMEBUFFER, self.gl.COLOR_ATTACHMENT1, self.gl.TEXTURE_2D, this.PositionRenderTexture, 0);
        this.ctx = canvas.getContext("2d");
        this.renderFrame = function() {
            if (dis.Enabled) {
				var CameraTransformInverse;
				if (self.canvas.width !== dis.Resolution[0] || self.canvas.height !== dis.Resolution[1]) {
					self.canvas.width = this.Resolution[0];
					self.canvas.height = this.Resolution[1];
				}
				var m = Math.max(self.canvas.width, self.canvas.height);
				var GT = this.Parrent.getGlobalTransform();
				var Transformz = [];
                if (dis.isVR) {
                    var EyeOffset = GT.matrix.Multiply(new self.Vector3([0.5,0,0]).Scale(-this.IPD));
                    var Transform0 = new self.transform(GT.traslation.Add(EyeOffset),GT.matrix).InverseTransform();
                    var Transform1 = new self.transform(GT.traslation.Subtract(EyeOffset),GT.matrix).InverseTransform();
					// EyeOffset = GT.matrix.Multiply(EyeOffset);
					Transformz = [GT.traslation.Add(EyeOffset),GT.traslation.Subtract(EyeOffset)];
                    CameraTransformInverse = [Transform0.ToMatrixTransposed3x4(),Transform1.ToMatrixTransposed3x4()];
					//CameraTransformInverse = [Transform1.ToMatrixTransposed3x4(),Transform0.ToMatrixTransposed3x4()];
                } else {
					Transformz = [GT.traslation];
                    CameraTransformInverse = [GT.InverseTransform().ToMatrixTransposed3x4()];
                }
				if (this.PostProcessing) {
					self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, this.FrameBuffer);
				} else {
					self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, null);
				}
				// self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, null);
                self.gl.clearColor(0.0,0.0,0.0, 1.0);
                self.gl.clear(self.gl.COLOR_BUFFER_BIT | self.gl.DEPTH_BUFFER_BIT);
                //self.canvas.width = resolution[0];
                //self.canvas.height = resolution[1];
				//console.log(RenderQue);
				var w = this.Resolution[0]/CameraTransformInverse.length;
				for (var k=0; k<CameraTransformInverse.length; k++) {
					self.gl.viewport(w*k, 0, w, this.Resolution[1]);
					for (var j=0; j<self.RenderQue.length; j++) {
						self.RenderQue[j].render(CameraTransformInverse[k],dis,Transformz[k].value);
					}
				}
				//MotionBlurPostProcessingShader
				self.gl.bindTexture(self.gl.TEXTURE_2D,this.ColorRenderTexture);
				self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.LINEAR_MIPMAP_LINEAR);
				self.gl.generateMipmap(self.gl.TEXTURE_2D);
				if (this.PostProcessing) {
					if (false) {
						self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, null);
						self.gl.activeTexture(self.gl.TEXTURE0);
						self.gl.bindTexture(self.gl.TEXTURE_2D,this.ColorRenderTexture);
						// self.gl.generateMipmap(self.gl.TEXTURE_2D);
						self.gl.activeTexture(self.gl.TEXTURE1);
						self.gl.bindTexture(self.gl.TEXTURE_2D,this.PositionRenderTexture);
						self.gl.useProgram(self.BloomPostProcessingShader.program);
						self.gl.uniform1i(self.DepthOfFeildProcessingShader.Uniforms.uColorTex, 0);
						// self.gl.uniform1i(self.DepthOfFeildProcessingShader.Uniforms.uPosTex, 1);
						// self.gl.uniform1f(self.DepthOfFeildProcessingShader.Uniforms.uFocalDistance.Location, 5);
						// self.gl.uniform1f(self.DepthOfFeildProcessingShader.Uniforms.uArperture.Location, 100);
						//{Name:'uFocalDistance',Type:'float'},{Name:'uArperture',Type:'float'}
						self.gl.viewport(0, 0, this.Resolution[0], this.Resolution[1]);
						self.gl.bindBuffer(self.gl.ELEMENT_ARRAY_BUFFER, self.VertexMapingIndexBuffer);
						self.gl.drawElements(self.gl.TRIANGLES,6,self.gl.UNSIGNED_SHORT,0);
					} else {
						self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, null);
						self.gl.activeTexture(self.gl.TEXTURE0);
						self.gl.bindTexture(self.gl.TEXTURE_2D,this.ColorRenderTexture);
						self.gl.activeTexture(self.gl.TEXTURE1);
						self.gl.bindTexture(self.gl.TEXTURE_2D,this.PositionRenderTexture);
						self.gl.useProgram(self.MotionBlurPostProcessingShader.program);
						var x3Mat = this.PreviousMat.Inverse().Multiply(GT.matrix).ToQuaternion().Slerp(self.IdentityQuaternion,0.5).ToMatrix().value;
						self.gl.uniformMatrix3fv(self.MotionBlurPostProcessingShader.Uniforms.uMotionMat.Location,true,x3Mat.flat());
						self.gl.viewport(0, 0, this.Resolution[0], this.Resolution[1]);
						var m = Math.max(this.Resolution[0],this.Resolution[1]);
						self.gl.uniform2fv(self.MotionBlurPostProcessingShader.Uniforms.uAspect.Location,[this.Resolution[0]/m,this.Resolution[1]/m]);
						self.gl.bindBuffer(self.gl.ELEMENT_ARRAY_BUFFER, self.VertexMapingIndexBuffer);
						self.gl.drawElements(self.gl.TRIANGLES,6,self.gl.UNSIGNED_SHORT,0);
					}
				}
				this.PreviousMat = GT.matrix;
                dis.ctx.clearRect(0,0,dis.canvas.width,dis.canvas.height);
                dis.ctx.drawImage(self.canvas,0,0,dis.canvas.width,dis.canvas.height);
            }
        }
    }
	this.Components.ReflectionProbe = function(res) {
        this.Enabled = true;
		res = res || 128;
		this.Resolution = res;
		this.Realtime = false;
		this.Cubemap = self.gl.createTexture();
		self.gl.bindTexture(self.gl.TEXTURE_CUBE_MAP, this.Cubemap);
		self.gl.texImage2D(self.gl.TEXTURE_CUBE_MAP_POSITIVE_X, 0, self.gl.RGBA32F, res, res, 0, self.gl.RGBA, self.gl.FLOAT, null);
		self.gl.texImage2D(self.gl.TEXTURE_CUBE_MAP_NEGATIVE_X, 0, self.gl.RGBA32F, res, res, 0, self.gl.RGBA, self.gl.FLOAT, null);
		self.gl.texImage2D(self.gl.TEXTURE_CUBE_MAP_POSITIVE_Y, 0, self.gl.RGBA32F, res, res, 0, self.gl.RGBA, self.gl.FLOAT, null);
		self.gl.texImage2D(self.gl.TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, self.gl.RGBA32F, res, res, 0, self.gl.RGBA, self.gl.FLOAT, null);
		self.gl.texImage2D(self.gl.TEXTURE_CUBE_MAP_POSITIVE_Z, 0, self.gl.RGBA32F, res, res, 0, self.gl.RGBA, self.gl.FLOAT, null);
		self.gl.texImage2D(self.gl.TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, self.gl.RGBA32F, res, res, 0, self.gl.RGBA, self.gl.FLOAT, null);
		self.gl.generateMipmap(self.gl.TEXTURE_CUBE_MAP);
		self.gl.texParameteri(self.gl.TEXTURE_CUBE_MAP, self.gl.TEXTURE_MIN_FILTER, self.gl.LINEAR_MIPMAP_LINEAR);
		//self.gl.texParameteri(self.gl.TEXTURE_CUBE_MAP, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
		this.RenderBuffer = self.gl.createFramebuffer();
		self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, this.RenderBuffer);
		this.RenderTexture = self.gl.createTexture();
		self.gl.bindTexture(self.gl.TEXTURE_2D, this.RenderTexture);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.CLAMP_TO_EDGE);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.CLAMP_TO_EDGE);
		self.gl.texImage2D(self.gl.TEXTURE_2D, 0, self.gl.RGBA32F, res, res, 0, self.gl.RGBA, self.gl.FLOAT, null);
		this.PositionRenderTexture = self.gl.createTexture();
		self.gl.bindTexture(self.gl.TEXTURE_2D, this.PositionRenderTexture);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.NEAREST);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.CLAMP_TO_EDGE);
		self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.CLAMP_TO_EDGE);
		self.gl.texImage2D(self.gl.TEXTURE_2D, 0, self.gl.RGBA32F, res, res, 0, self.gl.RGBA, self.gl.FLOAT, null);
		self.gl.drawBuffers([self.gl.COLOR_ATTACHMENT0,self.gl.COLOR_ATTACHMENT1]);
		self.gl.framebufferTexture2D(self.gl.FRAMEBUFFER, self.gl.COLOR_ATTACHMENT0, self.gl.TEXTURE_2D, this.RenderTexture, 0);
		self.gl.framebufferTexture2D(self.gl.FRAMEBUFFER, self.gl.COLOR_ATTACHMENT1, self.gl.TEXTURE_2D, this.PositionRenderTexture, 0);
		this.DepthBuffer = self.gl.createRenderbuffer();
		self.gl.bindRenderbuffer(self.gl.RENDERBUFFER, this.DepthBuffer);
		self.gl.renderbufferStorage(self.gl.RENDERBUFFER, self.gl.DEPTH_COMPONENT32F, res, res);
		self.gl.framebufferRenderbuffer(self.gl.FRAMEBUFFER, self.gl.DEPTH_ATTACHMENT, self.gl.RENDERBUFFER, this.DepthBuffer);
		this.CubemapFrameBuffer = self.gl.createFramebuffer();
		this.getData = function() {
			var p = this.Parrent.getGlobalTransform().traslation;
			var pl = p.value.slice();
			pl.push(Math.floor(Math.log2(this.Resolution)))
			return {Cubemap:this.Cubemap,Position:p,PosLod:pl};
		}
		this.Update = function() {
			if (this.Enabled && this.Realtime) {
				this.Render();
			}
		}
		this.Render = function() {
			if (this.Enabled) {
				self.gl.enable(self.gl.DEPTH_TEST);
				var Pos = this.Parrent.getGlobalTransform().traslation;
				for (var i=0; i<self.CubemapMatrixTransforms.length; i++) {
					self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, this.RenderBuffer);
					var CamTranfm = new self.transform(Pos.Oposite(),self.CubemapMatrixTransforms[i]).ToMatrix3x4();
					self.gl.clearColor(0.0,0.0,0.0, Infinity);
					self.gl.clear(self.gl.COLOR_BUFFER_BIT | self.gl.DEPTH_BUFFER_BIT);
					for (var j=0; j<self.RenderQue.length; j++) {
						var rp = self.RenderQue[j].ReflectionProbe;
						self.RenderQue[j].ReflectionProbe = new Array(4).fill({Cubemap:null,Distence:Infinity,Position:{value:[0,0,0]},PosLod:[0,0,0,0]});
						self.gl.viewport(0,0,this.Resolution,this.Resolution);
						self.RenderQue[j].render(CamTranfm,{FeildOfView:59,ClipDistance:0.2},Pos.value);
						self.RenderQue[j].ReflectionProbe = rp;
					}
					//ReflectionProbePostProcessingShader
					self.gl.useProgram(self.ReflectionProbePostProcessingShader.program);
					
					self.gl.activeTexture(self.gl.TEXTURE0);
					self.gl.bindTexture(self.gl.TEXTURE_2D,this.RenderTexture);
					self.gl.activeTexture(self.gl.TEXTURE1);
					self.gl.bindTexture(self.gl.TEXTURE_2D, this.PositionRenderTexture);
					
					self.gl.bindFramebuffer(self.gl.FRAMEBUFFER, this.CubemapFrameBuffer);
					self.gl.bindTexture(self.gl.TEXTURE_CUBE_MAP, this.Cubemap);
					self.gl.framebufferTexture2D(self.gl.FRAMEBUFFER, self.gl.COLOR_ATTACHMENT0, self.gl.TEXTURE_CUBE_MAP_POSITIVE_X+i, this.Cubemap, 0);
					self.gl.clearColor(0.0,0.0,0.0, Infinity);
					self.gl.clear(self.gl.COLOR_BUFFER_BIT | self.gl.DEPTH_BUFFER_BIT);
					self.gl.bindBuffer(self.gl.ELEMENT_ARRAY_BUFFER, self.VertexMapingIndexBuffer);
					self.gl.drawElements(self.gl.TRIANGLES,6,self.gl.UNSIGNED_SHORT,0);
				}
				self.gl.bindTexture(self.gl.TEXTURE_CUBE_MAP, this.Cubemap);
				self.gl.generateMipmap(self.gl.TEXTURE_CUBE_MAP);
			}
		}
	}
    this.Components.BoxCollider = function(p0,p1) {
        this.Enabled = true;
		p0 = p0 || new Engine.Vector3([0.5,0.5,0.5]);
		p1 = p1 || new Engine.Vector3([-0.5,-0.5,-0.5]);
		this.Min = p0.Min(p1);
		this.Max = p0.Max(p1);
		this.RayCastLocal = function(pos,dir,bothsides) {
            if (this.Enabled) {
				var d = new self.Vector3([1/dir.value[0],1/dir.value[1],1/dir.value[2]]);
				var t0 = this.Min.Subtract(pos).Multiply(d);
				var t1 = this.Max.Subtract(pos).Multiply(d);
				var tn = Math.max(Math.min(t0.value[0],t1.value[0]),Math.min(t0.value[1],t1.value[1]),Math.min(t0.value[2],t1.value[2]));
				var tx = Math.min(Math.max(t0.value[0],t1.value[0]),Math.max(t0.value[1],t1.value[1]),Math.max(t0.value[2],t1.value[2]));
				if (tn > tx) {
					return Infinity;
				}
				if (bothsides) {
					return tn;
				} else {
					if (tx < 0) {
						return Infinity;
					}
					return Math.max(tn,0);
				}
			} else {
				return Infinity;
			}
        }
        this.RayCastGlobal = function(pos,dir,bothsides) {
            if (this.Enabled) {
				var trsfm = this.Parrent.getGlobalTransform().InverseTransform();
				return this.RayCastLocal(trsfm.TransformPoint(pos),trsfm.TransformVector(dir).Normalize(),bothsides);
			} else {
				return Infinity;
			}
        }
		this.RayCastLocalData = function(pos,dir,bothsides) {
			var d = new self.Vector3([1/dir.value[0],1/dir.value[1],1/dir.value[2]]);
			var t0 = this.Min.Subtract(pos).Multiply(d);
			var t1 = this.Max.Subtract(pos).Multiply(d);
			var tn = Math.max(Math.min(t0.value[0],t1.value[0]),Math.min(t0.value[1],t1.value[1]),Math.min(t0.value[2],t1.value[2]));
			var tx = Math.min(Math.max(t0.value[0],t1.value[0]),Math.max(t0.value[1],t1.value[1]),Math.max(t0.value[2],t1.value[2]));
			return {Max:tx,Min:tn};
        }
        this.RayCastGlobalData = function(pos,dir,bothsides) {
			var trsfm = this.Parrent.getGlobalTransform().InverseTransform();
			return this.RayCastLocalData(trsfm.TransformPoint(pos),trsfm.TransformVector(dir).Normalize(),bothsides);
        }
		this.getColisionData = function(other) {
			if (this.Enabled && other.Enabled) {
				if (other instanceof self.Components.BoxCollider) {
					var trsfm0 = this.Parrent.getGlobalTransform();
					var trsfm1 = other.Parrent.getGlobalTransform();
					var trsfm2 = trsfm1.InverseTransform();
					var trsfm3 = trsfm0.InverseTransform();
					var OtherCenter = trsfm3.TransformPoint(trsfm1.TransformPoint(other.Max.Add(other.Min).Scale(0.5)));
					if (this.isLocalPointInside(OtherCenter)) {
						return {Coliding:true,GlobalPosition:OtherCenter,UnIntersect:trsfm0.TransformVector(OtherCenter.Normalize().Scale(this.RayCastLocal(new self.Vector3(),OtherCenter.Normalize(),true)))};
					} else {
						var closestPoint = null;
						var closestPointOther = null;
						var closestPointIdx = 0;
						var closestPointDist = Infinity;
						for (var i=0; i<8; i++) {
							var p0 = this.getVertex(i);
							// var p1 = trsfm3.TransformPoint(p0);
							var p1 = trsfm2.TransformPoint(trsfm0.TransformPoint(p0));
							var dist = other.distenceToLocalPoint(p1);
							if (dist < closestPointDist) {
								closestPoint = p0;
								closestPointDist = dist;
								closestPointOther = p1;
								closestPointIdx = i;
							}
						}
						// console.log(closestPoint,closestPointOther,closestPointIdx,closestPointDist);
						if (closestPointDist <= 0) {
							var center = other.Max.Add(other.Min).Scale(0.5);
							var size = other.Max.Subtract(other.Min).Scale(0.5);
							var poi = closestPointOther.Subtract(center);
							var mins = size.Subtract(poi.Abs());
							var min = Math.min(mins.value[0],mins.value[1],mins.value[2]);
							var isect = new self.Vector3();
							if (mins.value[0] == min) {
								isect.value[0] = poi.value[0] > 0 ? min : -min;
							}
							if (mins.value[1] == min) {
								isect.value[1] = poi.value[1] > 0 ? min : -min;
							}
							if (mins.value[2] == min) {
								isect.value[2] = poi.value[2] > 0 ? min : -min;
							}
							isect = trsfm1.TransformVector(isect);
							return {Coliding:true,GlobalPosition:trsfm1.TransformPoint(closestPointOther),UnIntersect:isect};
						} else {
							// return {Coliding:false};
							//var isect = [];//new self.Vector3();
							var pos = [];//new self.Vector3();
							// var count = 0;
							var dir = (new self.Vector3([closestPointIdx%2 == 1 ? -1 : 1,0,0]));
							var dir2 = trsfm2.TransformVector(trsfm0.TransformVector(dir));
							var dir2len = dir2.Length();
							dir2.ScaleThisBy(1/dir2len);
							var max = (this.Max.value[0]-this.Min.value[0])/dir2len;
							var rc = other.RayCastLocalData(closestPointOther,dir2);
							var rcdist = rc.Min > rc.Max || rc.Min < 0 ? Infinity : rc.Min;
							if (rcdist <= max) {
								// return {Coliding:true,GlobalPosition:trsfm0.TransformPoint(closestPoint),LocalPosition:closestPoint,UnIntersect:trsfm0.TransformVector(dir.Scale(rc-max))};
								pos.push(closestPointOther.Add(dir2.Scale(Math.min((rc.Min+rc.Max)*0.5,max))));
								//isect.push(dir.Scale((max-rc.Min)*dir2len));
							}
							dir = (new self.Vector3([0,Math.floor(closestPointIdx/2)%2 == 1 ? -1 : 1,0]));
							dir2 = trsfm2.TransformVector(trsfm0.TransformVector(dir));
							dir2len = dir2.Length();
							dir2.ScaleThisBy(1/dir2len);
							max = (this.Max.value[1]-this.Min.value[1])/dir2len;
							rc = other.RayCastLocalData(closestPointOther,dir2);
							rcdist = rc.Min > rc.Max || rc.Min < 0 ? Infinity : rc.Min;
							if (rcdist <= max) {
								// return {Coliding:true,GlobalPosition:trsfm0.TransformPoint(closestPoint),LocalPosition:closestPoint,UnIntersect:trsfm0.TransformVector(dir.Scale(rc-max))};
								pos.push(closestPointOther.Add(dir2.Scale(Math.min((rc.Min+rc.Max)*0.5,max))));
								//isect.push(dir.Scale((max-rc.Min)*dir2len));
							}
							dir = (new self.Vector3([0,0,Math.floor(closestPointIdx/4)%2 == 1 ? -1 : 1]));
							dir2 = trsfm2.TransformVector(trsfm0.TransformVector(dir));
							dir2len = dir2.Length();
							dir2.ScaleThisBy(1/dir2len);
							max = (this.Max.value[2]-this.Min.value[2])/dir2len;
							rc = other.RayCastLocalData(closestPointOther,dir2);
							rcdist = rc.Min > rc.Max || rc.Min < 0 ? Infinity : rc.Min;
							if (rcdist <= max) {
								// return {Coliding:true,GlobalPosition:trsfm0.TransformPoint(closestPoint),LocalPosition:closestPoint,UnIntersect:trsfm0.TransformVector(dir.Scale(rc-max))};
								pos.push(closestPointOther.Add(dir2.Scale(Math.min((rc.Min+rc.Max)*0.5,max))));
								//isect.push(dir.Scale((max-rc.Min)*dir2len));
							}
							if (pos.length > 0) {
								//var isct = trsfm0.TransformVector(isect.reduce(function(a,b){return a.Add(b)}));//.ScaleThisBy(1/isect.length));
								var ps = pos.reduce(function(a,b){return a.Add(b)}).ScaleThisBy(1/pos.length);
								//var ps = trsfm3.TransformPoint(trsfm1.TransformPoint(pos.reduce(function(a,b){return a.Add(b)}).ScaleThisBy(1/pos.length)));
								//var isct = trsfm0.TransformVector(OtherCenter.Normalize().Scale(this.RayCastLocal(new self.Vector3(),OtherCenter.Normalize(),true)));
								//var isct = trsfm1.TransformVector(ps.Normalize().Scale(other.RayCastLocal(new self.Vector3(),ps.Normalize(),true)));
								var center = other.Max.Add(other.Min).Scale(0.5);
								var size = other.Max.Subtract(other.Min).Scale(0.5);
								var poi = ps.Subtract(center);
								var mins = size.Subtract(poi.Abs());
								var min = Math.min(mins.value[0],mins.value[1],mins.value[2]);
								var isect = new self.Vector3();
								if (mins.value[0] == min) {
									isect.value[0] = poi.value[0] > 0 ? min : -min;
								}
								if (mins.value[1] == min) {
									isect.value[1] = poi.value[1] > 0 ? min : -min;
								}
								if (mins.value[2] == min) {
									isect.value[2] = poi.value[2] > 0 ? min : -min;
								}
								return {Coliding:true,GlobalPosition:trsfm1.TransformPoint(ps),UnIntersect:trsfm1.TransformVector(isect)};
								//return {Coliding:true,GlobalPosition:trsfm0.TransformPoint(closestPoint),UnIntersect:trsfm1.TransformVector(isect)};
								// return {Coliding:true,GlobalPosition:trsfm0.TransformPoint(closestPoint),UnIntersect:isct};
							} else {
								return {Coliding:false};
							}
						}
					}
				}
			} else {
				return {Coliding:false};
			}
		}
		this.isLocalPointInside = function(point) {
			if (this.Enabled) {
				return ((point.value[0] >= this.Min.value[0] && point.value[0] <= this.Max.value[0]) && (point.value[1] >= this.Min.value[1] && point.value[1] <= this.Max.value[1])) && (point.value[2] >= this.Min.value[2] && point.value[2] <= this.Max.value[2]);
			} else {
				return false;
			}
		}
		this.isGlobalPointInside = function(point) {
			if (this.Enabled) {
				var p = this.Parrent.getGlobalTransform().InverseTransformPoint(point);
				return ((p.value[0] >= this.Min.value[0] && p.value[0] <= this.Max.value[0]) && (p.value[1] >= this.Min.value[1] && p.value[1] <= this.Max.value[1])) && (p.value[2] >= this.Min.value[2] && p.value[2] <= this.Max.value[2]);
			} else {
				return false;
			}
		}
		this.distenceToLocalPoint = function(point) {
			var dist = point.Subtract(point.Max(this.Min).Min(this.Max)).Length();
			if (dist <= 0) {
				var midle = this.Min.Add(this.Max).Scale(0.5);
				var size = this.Max.Subtract(this.Min).Scale(0.5);
				var mins = point.Subtract(midle).Abs().Subtract(size);
				// var mins = size.Subtract(point.Subtract(midle).Abs());
				return Math.max(mins.value[0],mins.value[1],mins.value[2]);
			} else {
				return dist;
			}
		}
		this.distenceToGlobalPoint = function(point) {
			return this.distenceToLocalPoint(this.Parrent.getGlobalTransform().InverseTransformPoint(point));
		}
		this.getVertex = function(idx) {
			return new self.Vector3([
				idx%2 == 0 ? this.Min.value[0] : this.Max.value[0],
				Math.floor(idx/2)%2 == 0 ? this.Min.value[1] : this.Max.value[1],
				Math.floor(idx/4)%2 == 0 ? this.Min.value[2] : this.Max.value[2]
			]);
		}
    }
    this.Components.RigidBody = function() {
        this.Enabled = true;
        this.velocity = new self.Vector3();
        this.angularVelocity = new self.Quaternion();
		//this.angularVelocity = self.randomQuaternion().SlerpThisBy(self.IdentityQuaternion,0.1);
        this.gravity = new self.Vector3([0,-0.01,0]);
        this.ApplyForceToCenterOfMass = function(force) {
            this.velocity.AddToThis(force);
        }
		this.PullTo = function(point,force,damp) {
			var trsfm = this.Parrent.Parrent.getGlobalTransform();
			var p = trsfm.TransformPoint(this.Parrent.transform.traslation);
			this.velocity.AddToThis(trsfm.InverseTransformPoint(point.Subtract(p)).Scale(force)).ScaleThisBy(damp);
		}
        this.Drag = function(d) {
            this.velocity.ScaleThisBy(d);
			this.angularVelocity = self.IdentityQuaternion.Slerp(this.angularVelocity,d);
        }
		this.getLocalVelocityAtPoint = function(point) {
			var v = new self.Vector3([this.angularVelocity.value[0],this.angularVelocity.value[1],this.angularVelocity.value[2]]);
			return v.Cross(point).Add(this.velocity);
			//return point.Cross(v).Add(this.velocity);
		}
		this.getGlobalVelocityAtPoint = function(point) {
			var gt = this.Parrent.getGlobalTransform();
			return gt.TransformVector(this.getLocalVelocityAtPoint(gt.InverseTransformPoint(point)));
		}
		this.calculateContactForce = function(LocalPosition,LocalForce) {
			//LocalPosition = this.Parrent.transform.matrix.Multiply(LocalPosition);
			var Force = this.Parrent.transform.matrix.Multiply(LocalPosition.Scale(LocalForce.Dot(LocalPosition)/LocalPosition.LengthSquared()));
			if (isNaN(Force.value[0]) || isNaN(Force.value[1]) || isNaN(Force.value[2])) {
				Force = new self.Vector3();
			}
			// var Force = this.Parrent.transform.matrix.Multiply(LocalPosition.Normalize().Scale(LocalForce.Dot(LocalPosition.Normalize())));
			// var Force = this.Parrent.transform.matrix.Multiply(LocalForce.Scale(LocalForce.Normalize().Dot(LocalPosition.Normalize())));
			var Torqe = self.lookAtRotation(LocalPosition,LocalPosition.Add(LocalForce));
			return {Torqe:Torqe,Force:Force};
		}
        this.Update = function() {
            if (this.Enabled) {
				// for (var q=0; q<2; q++) {
					this.velocity.AddToThis(this.gravity);
					var thisParrentGlobalTransform = this.Parrent.Parrent.getGlobalTransform();
					var thisTransform = this.Parrent.transform;
					var thisGlobalTransform = thisParrentGlobalTransform.TransformTransform(thisTransform);
					var thisInverseTransform = thisTransform.InverseTransform();
					var thisInverseGlobalTransform = thisGlobalTransform.InverseTransform();
					var thisParrentInverseGlobalTransform = thisParrentGlobalTransform.InverseTransform();
					var unIntersect = new self.Vector3();
					var unIntersectMax = 0;
					var Torqe = new self.Quaternion();
					var Force = new self.Vector3();
					var Count = 0;
					//this.Parrent.transform.traslation.AddToThis(this.velocity.Scale(0.25));
					//this.Parrent.transform.matrix.MultiplyThisBy(self.IdentityQuaternion.Slerp(this.angularVelocity,0.25).ToMatrix());
					this.Parrent.transform.traslation.AddToThis(this.velocity);
					this.Parrent.transform.matrix.MultiplyThisBy(this.angularVelocity.ToMatrix());
					var selfColiders = [];
					for (var i=0; i<this.Parrent.Components.length; i++) {
						if (this.Parrent.Components[i] instanceof self.Components.BoxCollider) {
							selfColiders.push(this.Parrent.Components[i]);
						}
					}
					var dis = this;
					function checkCollisions(obj) {
						var isRidgidBody = false;
						for (var i=0; i<obj.Components.length; i++) {
							if (obj.Components[i] instanceof self.Components.RigidBody) {
								isRidgidBody = obj.Components[i];
								break;
							}
						}
						for (var i=0; i<obj.Components.length; i++) {
							if (obj.Components[i] instanceof self.Components.BoxCollider) {
								for (var j=0; j<selfColiders.length; j++) {
									//var data = obj.Components[i].getColisionData(selfColiders[j]);
									var data = selfColiders[j].getColisionData(obj.Components[i]);
									if (data.Coliding) {
										var p1 = thisParrentInverseGlobalTransform.TransformVector(data.GlobalPosition);
										var p2 = thisInverseGlobalTransform.TransformPoint(data.GlobalPosition);
										//if (!isRidgidBody) {
											// var p0 = dis.Parrent.transform.TransformVector(thisInverseGlobalTransform.TransformVector(data.UnIntersect));
											var p0 = thisParrentInverseGlobalTransform.TransformVector(data.UnIntersect);
											unIntersect.AddToThis(p0);
											//unIntersect.AddToThis(p0.Flatten(unIntersect.Normalize()));
											// var l = p0.LengthSquared();
											// if (l > unIntersectMax) {
											// 	unIntersect = p0;
											// 	unIntersectMax = l;
											// }
										//}
										var F = thisInverseGlobalTransform.TransformVector(data.UnIntersect);
										if (isRidgidBody) {
											var f = thisInverseGlobalTransform.TransformVector(isRidgidBody.getGlobalVelocityAtPoint(data.GlobalPosition));
											F.AddToThis(f).ScaleThisBy(0.5);
											//var Vol = f.Subtract(thisInverseTransform.TransformVector(dis.getLocalVelocityAtPoint(p1))).Scale(0.5);
											//var Vol = f.Subtract(dis.getLocalVelocityAtPoint(p2));
											//var Vol = new self.Vector3();
										}
										//} else {
											//var Vol = thisInverseTransform.TransformVector(dis.getLocalVelocityAtPoint(p1)).Oposite();
										//	var Vol = dis.getLocalVelocityAtPoint(p2).Oposite();
										//}
										var calced = dis.calculateContactForce(p2,F);
										Torqe.MultiplyThis(calced.Torqe);
										Force.AddToThis(calced.Force);
										Count++;
									}
								}
							}
						}
						for (var i=0; i<obj.Children.length; i++) {
							if (obj.Children[i] != dis.Parrent) {
								checkCollisions(obj.Children[i]);
							}
						}
					}
					checkCollisions(self.Root);
					Torqe.SlerpThisBy(self.IdentityQuaternion,1-(1/Count));
					Force.ScaleThisBy(1/Count);
					// unIntersect.ScaleThisBy(1/Count);
					if (Count > 0) {
						this.Drag(0.75);
						//this.angularVelocity.MultiplyThis(Torqe.Multiply(self.IdentityQuaternion.Slerp(this.angularVelocity.Conjugate(),0.25)));
						//this.angularVelocity.MultiplyThis(self.IdentityQuaternion.Slerp(this.angularVelocity.Conjugate(),0.25).Multiply(Torqe));
						//this.velocity.AddToThis(Force.Subtract(this.velocity.Scale(0.25)));
						// this.velocity.AddToThis(Force.Add(unIntersect));
						this.angularVelocity.MultiplyThis(Torqe);
						this.velocity.AddToThis(Force);
						//this.angularVelocity = Torqe;
						//this.velocity = Force;
						//this.Drag(Count > 3 ? 0.6 : 0.8);
						//this.Drag(0.8);
						this.Parrent.transform.traslation.AddToThis(unIntersect);
						// this.Parrent.transform.traslation.AddToThis(unIntersect);
					}
				// }
				this.Drag(0.99);
			}
        }
    }
	this.Components.DynamicBone = function(stiff,frict,damp) {
		this.Enabled = true;
		this.Prev = null;
		this.Default = null;
		this.Offset = self.IdentityQuaternion;
		this.Momentum = self.IdentityQuaternion;
		this.Stiffniss = stiff;
		this.Friction = frict;
		this.Damping = damp;
		
		//this.Point = new self.Vector3([0,0,0]);
        this.Update = function() {
			if (this.Enabled) {
				if (!this.Default) {
					this.Default = this.Parrent.transform.matrix.ToQuaternion();
				}
				if (!this.Prev) {
					this.Prev = this.Parrent.getGlobalTransform().matrix.ToQuaternion();
				}
				//var t0 = this.Parrent.Parrent.getGlobalTransform();
				//var t1 = t0.TransformTransform(this.Parrent.transform);
				var t1 = this.Parrent.Parrent.getGlobalTransform().matrix.ToQuaternion();
				var q0 = this.Prev.Multiply(t1.Conjugate());
				this.Momentum = this.Momentum.Multiply(q0);
				this.Offset = this.Offset.Multiply(q0.Conjugate());
				this.Momentum = this.Momentum.Slerp(this.Offset,-this.Stiffniss);
				this.Momentum = this.Momentum.Slerp(self.IdentityQuaternion,this.Friction);
				this.Offset = this.Offset.Multiply(this.Momentum);
				this.Offset = this.Offset.Slerp(self.IdentityQuaternion,this.Damping);
				//this.Change = this.Offset.Multiply(q1.Conjugate());
				if (isNaN(this.Offset.value[0])) {
					this.Momentum = self.IdentityQuaternion;
					this.Offset = self.IdentityQuaternion;
				}
				this.Parrent.setMatrix(this.Default.Multiply(this.Offset).ToMatrix());
				this.Prev = t1;
			}
		}
	}
	this.Components.CopyGlobalTransform = function(obj) {
		this.Enabled = true;
		this.Scorce = obj;
		this.Update = function() {
			if (this.Enabled) {
				this.Parrent.transform = this.Parrent.Parrent.getGlobalTransform().InverseTransformTransform(this.Scorce.getGlobalTransform());
			}
		}
	}
    this.Components.IKchain = function(Goal,Target) {
        this.Enabled = true;
        this.GoalObject = Goal;
        this.TargetObject = Target;
        this.Iterations = 16;
		this.Chain = [];
        this.setTarget = function(Target) {
            this.TargetObject = Target;
        }
        this.setGoal = function(Goal) {
            this.GoalObject = Goal;
        }
        this.Update = function() {
            if (this.Enabled) {
				var GoalGlobalTransform = this.GoalObject.getGlobalTransform();
                var GoalGlobalPoint = GoalGlobalTransform.traslation;
				//var StartPoint = this.Parrent.getGlobalTransform().traslation;
                var Chain = [];
                var CurrentObject = this.TargetObject;
				var p = CurrentObject.getGlobalTransform().traslation;
                for (var j=0; CurrentObject!==this.Parrent; j++) {
                    Chain.push(CurrentObject);
					//Chain.unshift(CurrentObject);
                    CurrentObject = CurrentObject.Parrent;
                    if (j >= 256) {
                        console.error("IK chain error: Your IK chain is either too long or Cant find the end of the chain");
                        return undefined;
                    }
                }
				for (var i=0; i<this.Iterations; i++) {
					for (var j=1; j<Chain.length; j++) {
						var trsnfm = Chain[j].getGlobalTransform().InverseTransform();
						var goal = trsnfm.TransformPoint(GoalGlobalPoint).NormalizeThis();
						var target = trsnfm.TransformPoint(this.TargetObject.getGlobalTransform().traslation).NormalizeThis();
						target.AddToThis(goal)
						var q = self.lookAtRotation(target,goal);
                        var m = Chain[j].transform.matrix.Multiply(q.ToMatrix());
                        for (var k=0; k<Chain[j].Components.length; k++) {
                            if (Object.values(self.Components.IKConstraints).includes(Chain[j].Components[k].constructor)) {
                                m = Chain[j].Components[k].ConstraintMatrix(m);
                            }
                        }
                        Chain[j].setMatrix(m);
					}
				}
				Chain[0].setMatrix(Chain[1].getGlobalTransform().matrix.Inverse().Multiply(GoalGlobalTransform.matrix.SetScale(1)));
            }
        }
    }
	this.Components.IKConstraints = {};
	this.Components.IKConstraints.ConstrainAxsisAngle = function(ax,angl,maxDeviation) {
		this.Enabled = true;
        this.Axis = ax;
        this.Agl = angl;
        this.Deav = maxDeviation;
        this.ConstraintQuaternion = function(q) {
            if (this.Enabled) {
                //var angle = q.getRotationArroundAxsis(this.Axis).Subtract(this.Agl);
				var angle = this.Agl.Subtract(q.getRotationArroundAxsis(this.Axis));
				var result = q.Clone();
				var d0 = angle.degrees();
				var d1 = this.Deav.degrees();
				if (d0 > d1) {
					//return self.axsisAngle(angle.Add(this.Deav),this.Axis);
					//return self.axsisAngle(angle.Subtract(this.Deav),this.Axis);
					
					result.MultiplyThisBy(self.axsisAngle(angle.Add(this.Deav),this.Axis));
				}
				if (d0 < -d1) {
					//return self.axsisAngle(angle.Subtract(this.Deav),this.Axis);
					//return self.axsisAngle(angle.Add(this.Deav),this.Axis);
					
					result.MultiplyThisBy(self.axsisAngle(angle.Subtract(this.Deav),this.Axis));
				}
				return result;
            } else {
                return q;
            }
        }
        this.ConstraintMatrix = function(m) {
            if (this.Enabled) {
                var q = m.ToQuaternion();
                var reslt = this.ConstraintQuaternion(q);
                return reslt.ToMatrix();
            } else {
                return m;
            }
        }
	}
    this.Components.IKConstraints.LockXaxisConstraint = function() {
        this.Enabled = true;
        this.ConstraintMatrix = function(m) {
            if (this.Enabled) {
                var x = new self.Vector3([1,0,0]);
				var y = new self.Vector3(m.value[1]).Flatten(x).Normalize();
				var z = x.Cross(y);
				return new self.Matrix3x3([x.value,y.value,z.value]);
            } else {
                return m;
            }
        }
    }
    this.Components.IKConstraints.LockYaxisConstraint = function() {
        this.Enabled = true;
        this.ConstraintMatrix = function(m) {
            if (this.Enabled) {
                var y = new self.Vector3([0,1,0]);
				var x = new self.Vector3(m.value[0]).Flatten(x).Normalize();
				var z = x.Cross(y);
				return new self.Matrix3x3([x.value,y.value,z.value]);
            } else {
                return m;
            }
        }
    }
	this.Components.IKConstraints.LockZaxisConstraint = function() {
        this.Enabled = true;
        this.ConstraintMatrix = function(m) {
            if (this.Enabled) {
                var z = new self.Vector3([0,0,1]);
				var x = new self.Vector3(m.value[0]).Flatten(z).Normalize();
				var y = x.Cross(z);
				return new self.Matrix3x3([x.value,y.value,z.value]);
            } else {
                return m;
            }
        }
    }
    this.Components.IKConstraints.XaxisRotationConstraint = function() {
        this.Enabled = true;
        this.ConstraintMatrix = function(m) {
            if (this.Enabled) {
                var x = new self.Vector3(m.value[0]);
				var x2 = new self.Vector3([1,0,0]);
                var q = self.lookAtRotation(x2,x);
				return q.ToMatrix();
            } else {
                return m;
            }
        }
    }
    this.Components.IKConstraints.YaxisRotationConstraint = function() {
        this.Enabled = true;
        this.ConstraintMatrix = function(m) {
            if (this.Enabled) {
                var y = new self.Vector3(m.value[0]);
				var y2 = new self.Vector3([0,1,0]);
                var q = self.lookAtRotation(y2,y);
				return q.ToMatrix();
            } else {
                return m;
            }
        }
    }
    this.Components.IKConstraints.ZaxisRotationConstraint = function() {
        this.Enabled = true;
        this.ConstraintMatrix = function(m) {
            if (this.Enabled) {
                var z = new self.Vector3(m.value[0]);
				var z2 = new self.Vector3([1,0,0]);
                var q = self.lookAtRotation(z2,z);
				return q.ToMatrix();
            } else {
                return m;
            }
        }
    }
	this.Components.XaxsisLookAt = function(obj) {
		this.Target = obj;
		this.Enabled = true;
		this.Update = function() {
			if (this.Enabled) {
				var v0 = new self.Vector3([1,0,0]);
				var t0 = this.Parrent.Parrent.getGlobalTransform()
				var t1 = t0.TransformTransform(this.Parrent.transform);
				var v1 = t1.traslation.Subtract(this.Target.getGlobalTransform().traslation);
				this.Parrent.setMatrix(t0.matrix.Inverse().Multiply(self.lookAtRotation(v1,v0).ToMatrix()));
			}
		}
	}
	this.Components.YaxsisLookAt = function(obj) {
		this.Target = obj;
		this.Enabled = true;
		this.Update = function() {
			if (this.Enabled) {
				var v0 = new self.Vector3([0,1,0]);
				var t0 = this.Parrent.Parrent.getGlobalTransform()
				var t1 = t0.TransformTransform(this.Parrent.transform);
				var v1 = t1.traslation.Subtract(this.Target.getGlobalTransform().traslation);
				this.Parrent.setMatrix(t0.matrix.Inverse().Multiply(self.lookAtRotation(v1,v0).ToMatrix()));
			}
		}
	}
	this.Components.ZaxsisLookAt = function(obj) {
		this.Target = obj;
		this.Enabled = true;
		this.Update = function() {
			if (this.Enabled) {
				var v0 = new self.Vector3([0,0,1]);
				var t0 = this.Parrent.Parrent.getGlobalTransform()
				var t1 = t0.TransformTransform(this.Parrent.transform);
				var v1 = t1.traslation.Subtract(this.Target.getGlobalTransform().traslation);
				this.Parrent.setMatrix(t0.matrix.Inverse().Multiply(self.lookAtRotation(v1,v0).ToMatrix()));
			}
		}
	}
    this.EulerAngles = function(x,y,z) {
        if (x || x == 0) {
            if (x instanceof self.Vector3) {
				y = Boolean(y);
				this.value = [new self.Angle(x.value[0],y),new self.Angle(x.value[1],y),new self.Angle(x.value[2],y)];
			} else {
				if (x instanceof self.Angle) {
					if (y instanceof self.Angle) {
						if (z instanceof self.Angle) {
							this.value = [x,y,z];
						} else {
							this.value = [x,y,new self.Angle(0)];
						}
					} else {
						this.value = [x,new self.Angle(0),new self.Angle(0)];
					}
				} else {
					if (x instanceof Array) {
						y = Boolean(y);
						this.value = [new self.Angle(x[0],y),new self.Angle(x[1],y),new self.Angle(x[2],y)];
					} else {
						this.value = [new self.Angle(0),new self.Angle(0),new self.Angle(0)];
					}
				}
			}
        } else {
            this.value = [new self.Angle(0),new self.Angle(0),new self.Angle(0)];
        }
		this.Add = function(a) {
			return new self.EulerAngles([this.value[0]+a.value[0],this.value[1]+a.value[1],this.value[2]+a.value[2]]);
		}
        this.Subtract = function(a) {
            return new self.EulerAngles([this.value[0]-a.value[0],this.value[1]-a.value[1],this.value[2]-a.value[2]]);
        }
        this.toMatrix = function() {
            var m = new self.Matrix3x3();
            m.rotateThisAroundXAxis(this.value[0]);
            m.rotateThisAroundYAxis(this.value[1]);
            m.rotateThisAroundZAxis(this.value[2]);
            return m;
        }
		this.toMatrixXZY = function() {
            var m = new self.Matrix3x3();
            m.rotateThisAroundXAxis(this.value[0]);
            m.rotateThisAroundZAxis(this.value[2]);
            m.rotateThisAroundYAxis(this.value[1]);
            return m;
        }
        this.toQuaternion = function() {
            var x = this.value[0].radians()/2;
            var y = this.value[1].radians()/2;
            var z = this.value[2].radians()/2;
            var c1 = Math.cos(x);
            var c2 = Math.cos(y);
            var c3 = Math.cos(z);
            var s1 = Math.sin(x);
            var s2 = Math.sin(y);
            var s3 = Math.sin(z);
            var w = c1*c2*c3-s1*s2*s3;
			x = s1*c2*c3+c1*s2*s3;
			y = c1*s2*c3-s1*c2*s3;
            z = s1*s2*c3+c1*c2*s3;
            return new self.Quaternion(w,x,y,z);
        }
        this.degrees = function() {
            return [this.value[0].degrees(),this.value[1].degrees(),this.value[2].degrees()]
        }
        this.radians = function() {
            return [this.value[0].radians(),this.value[1].radians(),this.value[2].radians()]
        }
    }
    this.Vector3 = function(x,y,z) {
        if (x || x === 0) {
            if (y || y === 0) {
                this.value = [x,y,z];
            } else {
                this.value = x;
            }
        } else {
            this.value = [0,0,0];
        }
        // console.log(this.value);
        this.__proto__.Set = function(a,b,c) {
            if (b) {
                this.value = [a,b,c];
            } else {
                if (a) {
                    if (a instanceof self.Vector3) {
                        this.value = a.value;
                    } else {
                        this.value = a;
                    }
                } else {
                    this.value = [0,0,0];
                }
            }
        }
        this.__proto__.Add = function(v) {
            return new self.Vector3([this.value[0]+v.value[0],this.value[1]+v.value[1],this.value[2]+v.value[2]]);
        }
		this.__proto__.Abs = function() {
            return new self.Vector3([Math.abs(this.value[0]),Math.abs(this.value[1]),Math.abs(this.value[2])]);
        }
        this.__proto__.AddToThis = function(v) {
            if (v instanceof self.Vector3) {
                this.value[0] += v.value[0];
                this.value[1] += v.value[1];
                this.value[2] += v.value[2];
            } else {
                this.value[0] += v[0];
                this.value[1] += v[1];
                this.value[2] += v[2];
            }
            return this;
        }
        this.__proto__.Subtract = function(v) {
            return new self.Vector3([this.value[0]-v.value[0],this.value[1]-v.value[1],this.value[2]-v.value[2]]);
        }
        this.__proto__.SubtractFrom = function(v) {
            return new self.Vector3([v.value[0]-this.value[0],v.value[1]-this.value[1],v.value[2]-this.value[2]]);
        }
        this.__proto__.SubtractThisBy = function(v) {
            this.value[0] -= v.value[0];
            this.value[1] -= v.value[1];
            this.value[2] -= v.value[2];
            return this;
        }
        this.__proto__.Multiply = function(v) {
            if (v instanceof self.Vector3) {
                return new self.Vector3([this.value[0]*v.value[0],this.value[1]*v.value[1],this.value[2]*v.value[2]]);
            } else {
                return new self.Vector3([this.value[0]*v,this.value[1]*v,this.value[2]*v]);
            }
        }
        this.__proto__.MultiplyThisBy = function(v) {
            this.value[0] *= v.value[0];
            this.value[1] *= v.value[1];
            this.value[2] *= v.value[2];
            return this;
        }
        this.__proto__.Lerp = function(v,t) {
            return new self.Vector3([this.value[0]+(v.value[0]-this.value[0])*t,this.value[1]+(v.value[1]-this.value[1])*t,this.value[2]+(v.value[2]-this.value[2])*t]);
        }
        this.__proto__.LerpThisBy = function(v,t) {
            this.value[0] += (v.value[0]-this.value[0])*t;
            this.value[1] += (v.value[1]-this.value[1])*t;
            this.value[2] += (v.value[2]-this.value[2])*t;
            return this;
        }
        this.__proto__.Scale = function(s) {
            return new self.Vector3([this.value[0]*s,this.value[1]*s,this.value[2]*s]);
        }
        this.__proto__.ScaleThisBy = function(s) {
            this.value[0] *= s;
            this.value[1] *= s;
            this.value[2] *= s;
            return this;
        }
        this.__proto__.Flatten = function(v) {
            //return v.Scale(v.Dot(this));
			var d = this.Dot(v);
			return this.Subtract(v.Scale(d));
        }
        this.__proto__.Divide = function(v) {
            if (v instanceof self.Vector3) {
                return new self.Vector3([this.value[0]/v.value[0],this.value[1]/v.value[1],this.value[2]/v.value[2]]);
            } else {
                return new self.Vector3([this.value[0]/v,this.value[1]/v,this.value[2]/v]);
            }
        }
        this.__proto__.DivideThisBy = function(v) {
            if (v instanceof self.Vector3) {
                this.value[0] /= v.value[0];
                this.value[1] /= v.value[1];
                this.value[2] /= v.value[2];
            } else {
                this.value[0] /= v;
                this.value[1] /= v;
                this.value[2] /= v;
            }
            return this;
        }
        this.__proto__.DivideByThis = function(v) {
            this.value = [v.value[0]/this.value[0],v.value[1]/this.value[1],v.value[2]/this.value[2]];
            return this;
        }
        this.__proto__.DivideFrom = function(v) {
            return new self.Vector3([v.value[0]/this.value[0],v.value[1]/this.value[1],v.value[2]/this.value[2]]);
        }
        this.__proto__.Dot = function(v) {
            return (this.value[0]*v.value[0])+(this.value[1]*v.value[1])+(this.value[2]*v.value[2]);
        }
        this.__proto__.Cross = function(v) {
            return new self.Vector3([(this.value[1]*v.value[2])-(this.value[2]*v.value[1]),(this.value[2]*v.value[0])-(this.value[0]*v.value[2]),(this.value[0]*v.value[1])-(this.value[1]*v.value[0])]);
        }
        this.__proto__.Length = function() {
            return Math.hypot(this.value[0],this.value[1],this.value[2]);
        }
        this.__proto__.LengthSquared = function() {
            return (this.value[0]*this.value[0])+(this.value[1]*this.value[1])+(this.value[2]*this.value[2]);
        }
        this.__proto__.DistenceTo = function(v) {
            if (v instanceof self.Vector3) {
                return Math.hypot(this.value[0]-v.value[0],this.value[1]-v.value[1],this.value[2]-v.value[2]);
            } else {
                return Math.hypot(this.value[0]-v[0],this.value[1]-v[1],this.value[2]-v[2]);
            }
        }
        this.__proto__.DistenceSquaredTo = function(v) {
            if (v instanceof self.Vector3) {
                var v0 = this.value[0]-v.value[0];
                var v1 = this.value[1]-v.value[1];
                var v2 = this.value[2]-v.value[2];
                return (v0*v0)+(v1*v1)+(v2*v2);
            } else {
                var v0 = this.value[0]-v[0];
                var v1 = this.value[1]-v[1];
                var v2 = this.value[2]-v[2];
                return (v0*v0)+(v1*v1)+(v2*v2);
            }
        }
        this.__proto__.Oposite = function() {
            return new self.Vector3([-this.value[0],-this.value[1],-this.value[2]]);
        }
		this.__proto__.OpositeThis = function() {
            this.value = [-this.value[0],-this.value[1],-this.value[2]];
			return this;
        }
        this.__proto__.InverseLength = function() {
            return this.Scale(1/this.Dot(this));
        }
		this.__proto__.Inverse = function() {
            return new self.Vector3([1/this.value[0],1/this.value[1],1/this.value[2]]);
        }
        this.__proto__.Normalize = function() {
            var l = Math.max(Math.hypot(this.value[0],this.value[1],this.value[2]),Number.EPSILON);
            return new self.Vector3([this.value[0]/l,this.value[1]/l,this.value[2]/l]);
        }
		this.__proto__.Min = function(v) {
			return new self.Vector3([Math.min(this.value[0],v.value[0]),Math.min(this.value[1],v.value[1]),Math.min(this.value[2],v.value[2])]);
		}
		this.__proto__.Max = function(v) {
			return new self.Vector3([Math.max(this.value[0],v.value[0]),Math.max(this.value[1],v.value[1]),Math.max(this.value[2],v.value[2])]);
		}
        this.__proto__.ToArray = function() {
            return this.value;
        }
        this.__proto__.NormalizeThis = function() {
            var l = Math.hypot(this.value[0],this.value[1],this.value[2]);
            this.value = [this.value[0]/l,this.value[1]/l,this.value[2]/l];
            return this;
        }
		this.__proto__.toArrayBuffer = function() {
			var arr = [];
			arr.push(new Uint8Array([86, 101, 99, 116, 111, 114, 51, 1]));
			var F32A = new Float32Array(this.value);
			arr.push(new Uint8Array(new Uint32Array([F32A.buffer.byteLength,0]).buffer));
			arr.push(new Uint8Array(F32A.buffer));
			var length = 0;
			for (var i=0; i<arr.length; i++) {
				length += arr[i].length;
			}
			arr[1].set(new Uint8Array(new Uint32Array([length-(arr[0].length+arr[1].length)]).buffer));
			var bufr = new Uint8Array(length);
			var idx = 0;
			for (var i=0; i<arr.length; i++) {
				bufr.set(arr[i],idx);
				idx += arr[i].length;
			}
			return bufr;
		}
    }
	this.Hue = function(h) {
		h *= 6;
		var r = Math.max(Math.min(Math.abs((h%6)-3)-1,1),0);
		var g = Math.max(Math.min(Math.abs(((h+2)%6)-3)-1,1),0);
		var b = Math.max(Math.min(Math.abs(((h+4)%6)-3)-1,1),0);
		return [r,g,b];
	}
	this.Wavelength = function(Wavelength) {
		var Red,Green,Blue;
		if((Wavelength >= 380) && (Wavelength < 440)) {
			Red = -(Wavelength - 440) / (440 - 380);
			Green = 0.0;
			Blue = 1.0;
		} else if((Wavelength >= 440) && (Wavelength < 490)) {
			Red = 0.0;
			Green = (Wavelength - 440) / (490 - 440);
			Blue = 1.0;
		} else if((Wavelength >= 490) && (Wavelength < 510)) {
			Red = 0.0;
			Green = 1.0;
			Blue = -(Wavelength - 510) / (510 - 490);
		} else if((Wavelength >= 510) && (Wavelength < 580)) {
			Red = (Wavelength - 510) / (580 - 510);
			Green = 1.0;
			Blue = 0.0;
		} else if((Wavelength >= 580) && (Wavelength < 645)) {
			Red = 1.0;
			Green = -(Wavelength - 645) / (645 - 580);
			Blue = 0.0;
		} else if((Wavelength >= 645) && (Wavelength < 781)) {
			Red = 1.0;
			Green = 0.0;
			Blue = 0.0;
		} else {
			Red = 0.0;
			Green = 0.0;
			Blue = 0.0;
		}
		var factor;
		if((Wavelength >= 380) && (Wavelength < 420)) {
			factor = 0.3 + 0.7 * (Wavelength - 380) / (420 - 380);
		} else if((Wavelength >= 420) && (Wavelength < 701)) {
			factor = 1.0;
		} else if((Wavelength >= 701) && (Wavelength < 781)) {
			factor = 0.3 + 0.7 * (780 - Wavelength) / (780 - 700);
		} else {
			factor = 0.0;
		}
		var gamma = 0.8;
		Red = (Red*factor)**gamma;
		Green = (Green*factor)**gamma;
		Blue = (Blue*factor)**gamma;
		return [Red,Green,Blue];
	}
    this.Quaternion = function(a,i,j,k) {
        if (a || a === 0) {
			if (i || i === 0) {
				if (i instanceof self.Vector3) {
					this.value = [i.value[0],i.value[1],i.value[2],a];
				} else {
					this.value = [i,j,k,a];
				}
			} else {
				this.value = a;
				// this.value = [a[1],a[2],a[3],a[0]];
			}
        } else {
            this.value = [0,0,0,1];
        }
        this.Clone = function() {
            return new self.Quaternion(this.value);
        }
        this.Multiply = function(v) {
			// var d = (this.value[0]*v.value[0])+(this.value[1]*v.value[1])+(this.value[2]*v.value[2])+(this.value[3]*v.value[3]);
			// if (d <= 0) {
			// 	v = {value:[-v.value[0],-v.value[1],-v.value[2],-v.value[3]]};
			// }
            return new self.Quaternion([
                (this.value[3]*v.value[0])+(this.value[0]*v.value[3])+(this.value[1]*v.value[2])-(this.value[2]*v.value[1]),
                (this.value[3]*v.value[1])+(this.value[1]*v.value[3])+(this.value[2]*v.value[0])-(this.value[0]*v.value[2]),
                (this.value[3]*v.value[2])+(this.value[2]*v.value[3])+(this.value[0]*v.value[1])-(this.value[1]*v.value[0]),
                (this.value[3]*v.value[3])-(this.value[0]*v.value[0])-(this.value[1]*v.value[1])-(this.value[2]*v.value[2])
            ]);
        }
        this.MultiplyThis = function(v) {
            this.value = [
                (this.value[3]*v.value[0])+(this.value[0]*v.value[3])+(this.value[1]*v.value[2])-(this.value[2]*v.value[1]),
                (this.value[3]*v.value[1])+(this.value[1]*v.value[3])+(this.value[2]*v.value[0])-(this.value[0]*v.value[2]),
                (this.value[3]*v.value[2])+(this.value[2]*v.value[3])+(this.value[0]*v.value[1])-(this.value[1]*v.value[0]),
                (this.value[3]*v.value[3])-(this.value[0]*v.value[0])-(this.value[1]*v.value[1])-(this.value[2]*v.value[2])
            ];
            return this;
        }
        this.MultiplyBy = function(v) {
            return new self.Quaternion([
                (v.value[0]*this.value[0])-(v.value[1]*this.value[1])-(v.value[2]*this.value[2])-(v.value[3]*this.value[3]),
                (v.value[0]*this.value[1])+(v.value[1]*this.value[0])+(v.value[2]*this.value[3])-(v.value[3]*this.value[2]),
                (v.value[0]*this.value[2])+(v.value[2]*this.value[0])+(v.value[3]*this.value[1])-(v.value[1]*this.value[3]),
                (v.value[0]*this.value[3])+(v.value[3]*this.value[0])+(v.value[1]*this.value[2])-(v.value[2]*this.value[1])
            ]);
        }
        this.MultiplyThisBy = function(v) {
            this.value = [
                (v.value[0]*this.value[0])-(v.value[1]*this.value[1])-(v.value[2]*this.value[2])-(v.value[3]*this.value[3]),
                (v.value[0]*this.value[1])+(v.value[1]*this.value[0])+(v.value[2]*this.value[3])-(v.value[3]*this.value[2]),
                (v.value[0]*this.value[2])+(v.value[2]*this.value[0])+(v.value[3]*this.value[1])-(v.value[1]*this.value[3]),
                (v.value[0]*this.value[3])+(v.value[3]*this.value[0])+(v.value[1]*this.value[2])-(v.value[2]*this.value[1])
            ];
            return this;
        }
        this.Add = function(v) {
            return new self.Quaternion([
                this.value[0]+v.value[0],
                this.value[1]+v.value[1],
                this.value[2]+v.value[2],
                this.value[3]+v.value[3]
            ]);
        }
		this.AddThis = function(v) {
            this.value = [this.value[0]+v.value[0],this.value[1]+v.value[1],this.value[2]+v.value[2],this.value[3]+v.value[3]];
			return this;
        }
        this.Conjugate = function() {
            return new self.Quaternion([-this.value[0],-this.value[1],-this.value[2],this.value[3]]);
        }
        this.Inverse = function() {
            return this.Conjugate().MultiplyBy(1/this.Length());
        }
        this.Length = function() {
            return Math.hypot(this.value[0],this.value[1],this.value[2],this.value[3]);
        }
        this.Normalize = function() {
            var l = this.Length();
            return new self.Quaternion([this.value[0]/l,this.value[1]/l,this.value[2]/l,this.value[3]/l]);
        }
		this.Scale = function(s) {
			return new self.Quaternion([this.value[0]*s,this.value[1]*s,this.value[2]*s,this.value[3]*s]);
		}
		this.ScaleThis = function(s) {
			this.value = [this.value[0]*s,this.value[1]*s,this.value[2]*s,this.value[3]*s];
			return this;
		}
		this.Weight = function(q1,a,b) {
			return this.Scale(a).AddThis(q1.Scale(b));
		}
		this.getRotationArroundAxsis = function(ax) {
			var v = new self.Vector3(this.value.slice(0,3));
			if (this.value[3] < 0) {
				v.OpositeThis();
			}
			//return new self.Angle(Math.acos(ax.Dot(v))*2,true);
			return new self.Angle(Math.asin(ax.Dot(v))*2,true);
		}
        this.ToEulerAngles = function() {
            var q = this.Normalize();
			var v = [q.value[3],q.value[0],q.value[1],q.value[2]];
			var v22 = (v[2]*v[2]);
			return new self.EulerAngles([
				Math.atan2(2*((v[0]*v[1])+(v[2]*v[3])),1-(2*((v[1]*v[1])+v22))),
				Math.asin(2*((v[0]*v[2])-(v[3]*v[1]))),
				Math.atan2(2*((v[0]*v[3])+(v[1]*v[2])),1-(2*(v22+(v[3]*v[3]))))
			],true);
        }
        this.ToMatrix = function() {
            var q = this.Normalize().value;
            var m = [[0,0,0],[0,0,0],[0,0,0]];
            m[0][0] = 1-2*(q[1]*q[1]+q[2]*q[2]);
            m[0][1] = 2*(q[0]*q[1]-q[2]*q[3]);
            m[0][2] = 2*(q[2]*q[0]+q[1]*q[3]);
            m[1][0] = 2*(q[0]*q[1]+q[2]*q[3]);
            m[1][1] = 1-2*(q[2]*q[2]+q[0]*q[0]);
            m[1][2] = 2*(q[1]*q[2]-q[0]*q[3]);
            m[2][0] = 2*(q[2]*q[0]-q[1]*q[3]);
            m[2][1] = 2*(q[1]*q[2]+q[0]*q[3]);
            m[2][2] = 1-2*(q[1]*q[1]+q[0]*q[0]);
            return new self.Matrix3x3(m);
        }
        this.Dot = function(v) {
            return (this.value[0]*v.value[0])+(this.value[1]*v.value[1])+(this.value[2]*v.value[2])+(this.value[3]*v.value[3]);
        }
		this.Nlerp = function(q,t) {
			return this.Weight(q,1-t,t).Normalize();
		}
        this.Slerp = function(q,t) {
            var d = this.Dot(q);
			if (d > 0.95 && (t <= 1 && t >= 0)) {
				return this.Nlerp(q,t);
			}
            var theta = Math.acos(d);
			var a = Math.sin((1-t)*theta);
			var b = Math.sin(t*theta);
			var div = Math.sin(theta);
			if (Math.abs(div) < 0.001) {
				return q;
			}
			return this.Weight(q,a,b).Scale(1/div);
        }
		this.SlerpThisBy = function(q,t) {
			this.value = this.Slerp(q,t).value;
			return this;
		}
    }
    this.lookAtRotation = function(a,v) {
        var d = a.Normalize();
		var e = v.Lerp(a,0.5).Normalize();
		var dot = d.Dot(e);
        return new self.Quaternion(dot,d.Cross(e));
    }
	this.axsisAngle = function(agl,v) {
		var vl2 = v.LengthSquared();
		var a = agl.radians()/2;
		var sin = Math.sin(a);
		var cos = Math.cos(a);
		if (vl2 <= 1.01 && vl2 >= 0.99) {
			return new self.Quaternion(cos,v.Scale(sin));
		} else {
			return new self.Quaternion(cos,v.Scale(sin/Math.sqrt(vl2)));
		}
	}
	this.randomPointInSphere = function() {
		var v = new self.Vector3([1,1,1]);
		while (v.Length() > 1) {
			v.value = [(Math.random()*2)-1,(Math.random()*2)-1,(Math.random()*2)-1];
		}
		return v;
	}
	this.randomPointOnSphere = function() {
		return self.randomPointInSphere().Normalize();
	}
    this.Matrix3x3 = function(m) {
        if (m) {
            this.value = m;
        } else {
            this.value = [[1,0,0],[0,1,0],[0,0,1]];
        }
        this.Multiply = function(mv) {
            if (mv instanceof self.Vector3) {
                return new self.Vector3([mv.value[0]*this.value[0][0]+mv.value[1]*this.value[0][1]+mv.value[2]*this.value[0][2],mv.value[0]*this.value[1][0]+mv.value[1]*this.value[1][1]+mv.value[2]*this.value[1][2],mv.value[0]*this.value[2][0]+mv.value[1]*this.value[2][1]+mv.value[2]*this.value[2][2]]);
            } else {
                if (mv instanceof Array) {
                    return new self.Vector3([mv[0]*this.value[0][0]+mv[1]*this.value[0][1]+mv[2]*this.value[0][2],mv[0]*this.value[1][0]+mv[1]*this.value[1][1]+mv[2]*this.value[1][2],mv[0]*this.value[2][0]+mv[1]*this.value[2][1]+mv[2]*this.value[2][2]]);
                } else {
                    return new self.Matrix3x3([
                        [this.value[0][0]*mv.value[0][0]+this.value[0][1]*mv.value[1][0]+this.value[0][2]*mv.value[2][0],this.value[0][0]*mv.value[0][1]+this.value[0][1]*mv.value[1][1]+this.value[0][2]*mv.value[2][1],this.value[0][0]*mv.value[0][2]+this.value[0][1]*mv.value[1][2]+this.value[0][2]*mv.value[2][2]],
                        [this.value[1][0]*mv.value[0][0]+this.value[1][1]*mv.value[1][0]+this.value[1][2]*mv.value[2][0],this.value[1][0]*mv.value[0][1]+this.value[1][1]*mv.value[1][1]+this.value[1][2]*mv.value[2][1],this.value[1][0]*mv.value[0][2]+this.value[1][1]*mv.value[1][2]+this.value[1][2]*mv.value[2][2]],
                        [this.value[2][0]*mv.value[0][0]+this.value[2][1]*mv.value[1][0]+this.value[2][2]*mv.value[2][0],this.value[2][0]*mv.value[0][1]+this.value[2][1]*mv.value[1][1]+this.value[2][2]*mv.value[2][1],this.value[2][0]*mv.value[0][2]+this.value[2][1]*mv.value[1][2]+this.value[2][2]*mv.value[2][2]]
                    ]);
                }
            }
        }
		this.Clone = function() {
			return new self.Matrix3x3(this.value);
		}
        this.ThatMultiplyThis = function(mv) {
            this.value = [
				[this.value[0][0]*mv.value[0][0]+this.value[1][0]*mv.value[0][1]+this.value[2][0]*mv.value[0][2],this.value[0][0]*mv.value[1][0]+this.value[1][0]*mv.value[1][1]+this.value[2][0]*mv.value[2][1],this.value[0][0]*mv.value[2][0]+this.value[1][0]*mv.value[2][1]+this.value[2][0]*mv.value[2][2]],
                [this.value[0][1]*mv.value[0][0]+this.value[1][1]*mv.value[0][1]+this.value[2][1]*mv.value[0][2],this.value[0][1]*mv.value[1][0]+this.value[1][1]*mv.value[1][1]+this.value[2][1]*mv.value[2][1],this.value[0][1]*mv.value[2][0]+this.value[1][1]*mv.value[2][1]+this.value[2][1]*mv.value[2][2]],
                [this.value[0][2]*mv.value[0][0]+this.value[1][2]*mv.value[0][1]+this.value[2][2]*mv.value[0][2],this.value[0][2]*mv.value[1][0]+this.value[1][2]*mv.value[1][1]+this.value[2][2]*mv.value[2][1],this.value[0][2]*mv.value[2][0]+this.value[1][2]*mv.value[2][1]+this.value[2][2]*mv.value[2][2]]
            ];
            return this;
        }
        this.MultiplyThisBy = function(mv) {
            this.value = [
                [this.value[0][0]*mv.value[0][0]+this.value[0][1]*mv.value[1][0]+this.value[0][2]*mv.value[2][0],this.value[0][0]*mv.value[0][1]+this.value[0][1]*mv.value[1][1]+this.value[0][2]*mv.value[2][1],this.value[0][0]*mv.value[0][2]+this.value[0][1]*mv.value[1][2]+this.value[0][2]*mv.value[2][2]],
                [this.value[1][0]*mv.value[0][0]+this.value[1][1]*mv.value[1][0]+this.value[1][2]*mv.value[2][0],this.value[1][0]*mv.value[0][1]+this.value[1][1]*mv.value[1][1]+this.value[1][2]*mv.value[2][1],this.value[1][0]*mv.value[0][2]+this.value[1][1]*mv.value[1][2]+this.value[1][2]*mv.value[2][2]],
                [this.value[2][0]*mv.value[0][0]+this.value[2][1]*mv.value[1][0]+this.value[2][2]*mv.value[2][0],this.value[2][0]*mv.value[0][1]+this.value[2][1]*mv.value[1][1]+this.value[2][2]*mv.value[2][1],this.value[2][0]*mv.value[0][2]+this.value[2][1]*mv.value[1][2]+this.value[2][2]*mv.value[2][2]]
            ];
			return this;
        }
        this.Transpose = function() {
            return new self.Matrix3x3([[this.value[0][0],this.value[1][0],this.value[2][0]],[this.value[0][1],this.value[1][1],this.value[2][1]],[this.value[0][2],this.value[1][2],this.value[2][2]]]);
        }
        this.Inverse = function() {
            var m = this.value;
            var v0 = ((m[1][1]*m[2][2])-(m[1][2]*m[2][1]));
            var v1 = ((m[1][2]*m[2][0])-(m[1][0]*m[2][2]));
            var v2 = ((m[1][0]*m[2][1])-(m[1][1]*m[2][0]));
            var det = (m[0][0]*v0)+(m[0][1]*v1)+(m[0][2]*v2);
            return new self.Matrix3x3([
                [v0/det,((m[0][2]*m[2][1])-(m[0][1]*m[2][2]))/det,((m[0][1]*m[1][2])-(m[0][2]*m[1][1]))/det],
                [v1/det,((m[0][0]*m[2][2])-(m[0][2]*m[2][0]))/det,((m[0][2]*m[1][0])-(m[0][0]*m[1][2]))/det],
                [v2/det,((m[0][1]*m[2][0])-(m[0][0]*m[2][1]))/det,((m[0][0]*m[1][1])-(m[0][1]*m[1][0]))/det],
            ]);
        }
        this.Add = function(mv) {
            return new self.Matrix3x3([
                [this.value[0][0]+mv.value[0][0],this.value[0][1]+mv.value[0][1],this.value[0][2]+mv.value[0][2]],
                [this.value[1][0]+mv.value[1][0],this.value[1][1]+mv.value[1][1],this.value[1][2]+mv.value[1][2]],
                [this.value[2][0]+mv.value[2][0],this.value[2][1]+mv.value[2][1],this.value[2][2]+mv.value[2][2]]
            ]);
        }
		this.AddToThis = function(mv) {
            this.value = [
                [this.value[0][0]+mv.value[0][0],this.value[0][1]+mv.value[0][1],this.value[0][2]+mv.value[0][2]],
                [this.value[1][0]+mv.value[1][0],this.value[1][1]+mv.value[1][1],this.value[1][2]+mv.value[1][2]],
                [this.value[2][0]+mv.value[2][0],this.value[2][1]+mv.value[2][1],this.value[2][2]+mv.value[2][2]]
            ];
			return this;
        }
        this.Determinant = function() {
            var m = this.value;
            return ((m[0][0]*((m[1][1]*m[2][2])-(m[1][2]*m[2][1])))+(m[0][1]*((m[1][2]*m[2][0])-(m[1][0]*m[2][2])))+(m[0][2]*((m[1][0]*m[2][1])-(m[1][1]*m[2][0]))));
        }
        this.Scale = function(s) {
            return new self.Matrix3x3([[this.value[0][0]*s,this.value[0][1]*s,this.value[0][2]*s],[this.value[1][0]*s,this.value[1][1]*s,this.value[1][2]*s],[this.value[2][0]*s,this.value[2][1]*s,this.value[2][2]*s]]);
        }
        this.ScaleXYZ = function(x,y,z) {
            return new self.Matrix3x3([[this.value[0][0]*x,this.value[0][1]*y,this.value[0][2]*z],[this.value[1][0]*x,this.value[1][1]*y,this.value[1][2]*z],[this.value[2][0]*x,this.value[2][1]*y,this.value[2][2]*z]]);
        }
        this.ScaleThisBy = function(s) {
            this.value = [[this.value[0][0]*s,this.value[0][1]*s,this.value[0][2]*s],[this.value[1][0]*s,this.value[1][1]*s,this.value[1][2]*s],[this.value[2][0]*s,this.value[2][1]*s,this.value[2][2]*s]];
			return this;
        }
        this.YaxsisLookAt = function(p) {
			var x = new self.Vector3(this.value[0]);
			var y = new self.Vector3(this.value[1]);
			var z = new self.Vector3(this.value[2]);
			p = p.Normalize();
			x = y.Cross(p).Normalize();
            z = y.Cross(x).Normalize();
            var yn = p.Normalize().Add(y).Normalize();
            this.value = [
                x.value,
                y.value,
                z.value
            ];
        }
        this.SetScale = function(s) {
            var l = Math.hypot(this.value[0][0],this.value[0][1],this.value[0][2]);
            this.value[0][0] = s*this.value[0][0]/l;
            this.value[0][1] = s*this.value[0][1]/l;
            this.value[0][2] = s*this.value[0][2]/l;
            l = Math.hypot(this.value[1][0],this.value[1][1],this.value[1][2]);
            this.value[1][0] = s*this.value[1][0]/l;
            this.value[1][1] = s*this.value[1][1]/l;
            this.value[1][2] = s*this.value[1][2]/l;
            l = Math.hypot(this.value[2][0],this.value[2][1],this.value[2][2]);
            this.value[2][0] = s*this.value[2][0]/l;
            this.value[2][1] = s*this.value[2][1]/l;
            this.value[2][2] = s*this.value[2][2]/l;
			return this;
        }
        this.GetScale = function() {
            var l0 = Math.hypot(this.value[0][0],this.value[0][1],this.value[0][2]);
            var l1 = Math.hypot(this.value[1][0],this.value[1][1],this.value[1][2]);
            var l2 = Math.hypot(this.value[2][0],this.value[2][1],this.value[2][2]);
            return Math.max(l0,l1,l2);
        }
        this.FixShearing = function() {
            var v0 = new self.Vector3(this.value[0]);
            var v1 = new self.Vector3(this.value[1]);
            var v2 = new self.Vector3(this.value[2]);
            this.value[2] = v0.Cross(v1).value;
            this.value[1] = v2.Cross(v0).value;
        }
        this.PhysxRotate = function(v) {
            var l = this.GetScale();
            var v0 = new self.Vector3(this.value[0]);
            v0 = v0.AddToThis(v0.Cross(v));
            var v1 = new self.Vector3(this.value[1]);
            v1 = v1.AddToThis(v1.Cross(v));
            var v2 = new self.Vector3(this.value[2]);
            v2 = v2.AddToThis(v2.Cross(v));
            this.value[0] = v0.value;
            this.value[1] = v1.value;
            this.value[2] = v2.value;
            this.FixShearing();
            this.SetScale(l);
        }
        this.ToEulerAngles = function() {
            var x = Math.atan2(this.value[2][1],this.value[2][2]);
            var y = Math.atan2(-this.value[2][0],Math.hypot(this.value[2][1],this.value[2][2]));
            var z = Math.atan2(this.value[1][0],this.value[0][0]);
            return new self.EulerAngles([x,y,z],true);
        }
        this.ToQuaternion = function() {
			var x = new self.Vector3([1,0,0]);
			var y = new self.Vector3([0,1,0]);
			var z = new self.Vector3([0,0,1]);
            var Mx = new self.Vector3([this.value[0][0],this.value[1][0],this.value[2][0]]).Lerp(x,0.5).Normalize();
			var My = new self.Vector3([this.value[0][1],this.value[1][1],this.value[2][1]]).Lerp(y,0.5).Normalize();
			var Mz = new self.Vector3([this.value[0][2],this.value[1][2],this.value[2][2]]).Lerp(z,0.5).Normalize();
			var ax = x.Cross(Mx).Add(y.Cross(My)).Add(z.Cross(Mz)).Scale(0.5);
			return new self.Quaternion(Math.sqrt(1-ax.LengthSquared()),ax).Normalize();
        }
        this.rotateAroundXAxis = function(agl) {
			var a = agl.radians();
            return this.Multiply(new self.Matrix3x3([[1,0,0],[0,Math.cos(a),-Math.sin(a)],[0,Math.sin(a),Math.cos(a)]]));
        }
        this.rotateThisAroundXAxis = function(agl) {
			var a = agl.radians();
            var m = [[1,0,0],[0,Math.cos(a),-Math.sin(a)],[0,Math.sin(a),Math.cos(a)]];
            this.value = [
                [m[0][0]*this.value[0][0]+m[0][1]*this.value[1][0]+m[0][2]*this.value[2][0],m[0][0]*this.value[0][1]+m[0][1]*this.value[1][1]+m[0][2]*this.value[2][1],m[0][0]*this.value[0][2]+m[0][1]*this.value[1][2]+m[0][2]*this.value[2][2]],
                [m[1][0]*this.value[0][0]+m[1][1]*this.value[1][0]+m[1][2]*this.value[2][0],m[1][0]*this.value[0][1]+m[1][1]*this.value[1][1]+m[1][2]*this.value[2][1],m[1][0]*this.value[0][2]+m[1][1]*this.value[1][2]+m[1][2]*this.value[2][2]],
                [m[2][0]*this.value[0][0]+m[2][1]*this.value[1][0]+m[2][2]*this.value[2][0],m[2][0]*this.value[0][1]+m[2][1]*this.value[1][1]+m[2][2]*this.value[2][1],m[2][0]*this.value[0][2]+m[2][1]*this.value[1][2]+m[2][2]*this.value[2][2]],
            ];
        }
        this.rotateThisAroundThisXAxis = function(agl) {
			var a = agl.radians();
            var m = [[1,0,0],[0,Math.cos(a),-Math.sin(a)],[0,Math.sin(a),Math.cos(a)]];
            this.value = [
                [this.value[0][0]*m[0][0]+this.value[0][1]*m[1][0]+this.value[0][2]*m[2][0],this.value[0][0]*m[0][1]+this.value[0][1]*m[1][1]+this.value[0][2]*m[2][1],this.value[0][0]*m[0][2]+this.value[0][1]*m[1][2]+this.value[0][2]*m[2][2]],
                [this.value[1][0]*m[0][0]+this.value[1][1]*m[1][0]+this.value[1][2]*m[2][0],this.value[1][0]*m[0][1]+this.value[1][1]*m[1][1]+this.value[1][2]*m[2][1],this.value[1][0]*m[0][2]+this.value[1][1]*m[1][2]+this.value[1][2]*m[2][2]],
                [this.value[2][0]*m[0][0]+this.value[2][1]*m[1][0]+this.value[2][2]*m[2][0],this.value[2][0]*m[0][1]+this.value[2][1]*m[1][1]+this.value[2][2]*m[2][1],this.value[2][0]*m[0][2]+this.value[2][1]*m[1][2]+this.value[2][2]*m[2][2]]
            ];
        }
        this.rotateAroundYAxis = function(agl) {
			var a = agl.radians();
            return this.Multiply(new self.Matrix3x3([[Math.cos(a),0,Math.sin(a)],[0,1,0],[-Math.sin(a),0,Math.cos(a)]]));
        }
        this.rotateThisAroundYAxis = function(agl) {
			var a = agl.radians();
            var m = [[Math.cos(a),0,Math.sin(a)],[0,1,0],[-Math.sin(a),0,Math.cos(a)]];
            this.value = [
                [m[0][0]*this.value[0][0]+m[0][1]*this.value[1][0]+m[0][2]*this.value[2][0],m[0][0]*this.value[0][1]+m[0][1]*this.value[1][1]+m[0][2]*this.value[2][1],m[0][0]*this.value[0][2]+m[0][1]*this.value[1][2]+m[0][2]*this.value[2][2]],
                [m[1][0]*this.value[0][0]+m[1][1]*this.value[1][0]+m[1][2]*this.value[2][0],m[1][0]*this.value[0][1]+m[1][1]*this.value[1][1]+m[1][2]*this.value[2][1],m[1][0]*this.value[0][2]+m[1][1]*this.value[1][2]+m[1][2]*this.value[2][2]],
                [m[2][0]*this.value[0][0]+m[2][1]*this.value[1][0]+m[2][2]*this.value[2][0],m[2][0]*this.value[0][1]+m[2][1]*this.value[1][1]+m[2][2]*this.value[2][1],m[2][0]*this.value[0][2]+m[2][1]*this.value[1][2]+m[2][2]*this.value[2][2]],
            ];
        }
        this.rotateThisAroundThisYAxis = function(agl) {
			var a = agl.radians();
            var m = [[Math.cos(a),0,Math.sin(a)],[0,1,0],[-Math.sin(a),0,Math.cos(a)]];
            this.value = [
                [this.value[0][0]*m[0][0]+this.value[0][1]*m[1][0]+this.value[0][2]*m[2][0],this.value[0][0]*m[0][1]+this.value[0][1]*m[1][1]+this.value[0][2]*m[2][1],this.value[0][0]*m[0][2]+this.value[0][1]*m[1][2]+this.value[0][2]*m[2][2]],
                [this.value[1][0]*m[0][0]+this.value[1][1]*m[1][0]+this.value[1][2]*m[2][0],this.value[1][0]*m[0][1]+this.value[1][1]*m[1][1]+this.value[1][2]*m[2][1],this.value[1][0]*m[0][2]+this.value[1][1]*m[1][2]+this.value[1][2]*m[2][2]],
                [this.value[2][0]*m[0][0]+this.value[2][1]*m[1][0]+this.value[2][2]*m[2][0],this.value[2][0]*m[0][1]+this.value[2][1]*m[1][1]+this.value[2][2]*m[2][1],this.value[2][0]*m[0][2]+this.value[2][1]*m[1][2]+this.value[2][2]*m[2][2]]
            ];
        }
        this.rotateAroundZAxis = function(agl) {
			var a = agl.radians();
            return this.Multiply(new self.Matrix3x3([[Math.cos(a),-Math.sin(a),0],[Math.sin(a),Math.cos(a),0],[0,0,1]]));
        }
        this.rotateThisAroundZAxis = function(agl) {
			var a = agl.radians();
            var m = [[Math.cos(a),-Math.sin(a),0],[Math.sin(a),Math.cos(a),0],[0,0,1]];
            this.value = [
                [m[0][0]*this.value[0][0]+m[0][1]*this.value[1][0]+m[0][2]*this.value[2][0],m[0][0]*this.value[0][1]+m[0][1]*this.value[1][1]+m[0][2]*this.value[2][1],m[0][0]*this.value[0][2]+m[0][1]*this.value[1][2]+m[0][2]*this.value[2][2]],
                [m[1][0]*this.value[0][0]+m[1][1]*this.value[1][0]+m[1][2]*this.value[2][0],m[1][0]*this.value[0][1]+m[1][1]*this.value[1][1]+m[1][2]*this.value[2][1],m[1][0]*this.value[0][2]+m[1][1]*this.value[1][2]+m[1][2]*this.value[2][2]],
                [m[2][0]*this.value[0][0]+m[2][1]*this.value[1][0]+m[2][2]*this.value[2][0],m[2][0]*this.value[0][1]+m[2][1]*this.value[1][1]+m[2][2]*this.value[2][1],m[2][0]*this.value[0][2]+m[2][1]*this.value[1][2]+m[2][2]*this.value[2][2]],
            ]
        }
        this.rotateThisAroundThisZAxis = function(agl) {
			var a = agl.radians();
            var m = [[Math.cos(a),-Math.sin(a),0],[Math.sin(a),Math.cos(a),0],[0,0,1]];
            this.value = [
                [this.value[0][0]*m[0][0]+this.value[0][1]*m[1][0]+this.value[0][2]*m[2][0],this.value[0][0]*m[0][1]+this.value[0][1]*m[1][1]+this.value[0][2]*m[2][1],this.value[0][0]*m[0][2]+this.value[0][1]*m[1][2]+this.value[0][2]*m[2][2]],
                [this.value[1][0]*m[0][0]+this.value[1][1]*m[1][0]+this.value[1][2]*m[2][0],this.value[1][0]*m[0][1]+this.value[1][1]*m[1][1]+this.value[1][2]*m[2][1],this.value[1][0]*m[0][2]+this.value[1][1]*m[1][2]+this.value[1][2]*m[2][2]],
                [this.value[2][0]*m[0][0]+this.value[2][1]*m[1][0]+this.value[2][2]*m[2][0],this.value[2][0]*m[0][1]+this.value[2][1]*m[1][1]+this.value[2][2]*m[2][1],this.value[2][0]*m[0][2]+this.value[2][1]*m[1][2]+this.value[2][2]*m[2][2]]
            ];
        }
    }
	//this.CubemapMatrixTransforms = [new self.Matrix3x3([[0,0,1],[0,1,0],[-1,0,0]]),new self.Matrix3x3([[0,0,-1],[0,1,0],[1,0,0]]),new self.Matrix3x3([[-1,0,0],[0,0,-1],[0,-1,0]]),new self.Matrix3x3([[-1,0,0],[0,0,1],[0,1,0]]),new self.Matrix3x3([[-1,0,0],[0,1,0],[0,0,-1]]),new self.Matrix3x3([[1,0,0],[0,1,0],[0,0,1]])];
	this.CubemapMatrixTransforms = [new self.Matrix3x3([[0,0,1],[0,1,0],[-1,0,0]]),new self.Matrix3x3([[0,0,-1],[0,1,0],[1,0,0]]),new self.Matrix3x3([[-1,0,0],[0,0,-1],[0,-1,0]]),new self.Matrix3x3([[-1,0,0],[0,0,1],[0,1,0]]),new self.Matrix3x3([[-1,0,0],[0,1,0],[0,0,-1]]),new self.Matrix3x3([[1,0,0],[0,1,0],[0,0,1]])];
    this.Matrix3x4 = function(m) {
        if (m) {
            this.value = m;
        } else {
            this.value = [[1,0,0,0],[0,1,0,0],[0,0,1,0]];
        }
    }
    this.weightedTransform = function(m,w) {
        var Mat = m[0].matrix.Scale(w[0]).Add(m[1].matrix.Scale(w[1])).Add(m[2].matrix.Scale(w[2])).Add(m[3].matrix.Scale(w[3]));
        var Vect = m[0].traslation.Scale(w[0]).Add(m[1].traslation.Scale(w[1])).Add(m[2].traslation.Scale(w[2])).Add(m[3].traslation.Scale(w[3]));
        return new self.transform(Vect,Mat);
    }
    this.transform = function(pos,mat) {
        if (mat) {
            if (mat instanceof self.Matrix3x3) {
                this.matrix = mat;
            } else {
                this.matrix = new self.Matrix3x3(mat);
            }
        } else {
            this.matrix = new self.Matrix3x3();
        }
        if (pos) {
            if (pos instanceof self.Vector3) {
                this.traslation = pos;
            } else {
                this.traslation = new self.Vector3(pos);
            }
        } else {
            this.traslation = new self.Vector3();
        }
		this.Clone = function() {
			return new self.transform(new self.Vector3(this.traslation.value), new self.Matrix3x3(this.matrix.value));
		}
        this.TransformPoint = function(point) {
            return this.matrix.Multiply(point).Add(this.traslation);
        }
        this.TransformVector = function(vector) {
            return this.matrix.Multiply(vector);
        }
        this.TransformTransform = function(trnsf) {
            return new self.transform(this.traslation.Add(this.matrix.Multiply(trnsf.traslation)),this.matrix.Multiply(trnsf.matrix));
        }
        this.InverseTransformPoint = function(point) {
            var Minv = this.matrix.Inverse();
            return Minv.Multiply(point).Subtract(Minv.Multiply(this.traslation));
        }
        this.InverseTransformVector = function(vector) {
            return this.matrix.Inverse().Multiply(vector);
        }
        this.InverseTransformTransform = function(trnsf) {
            var Minv = this.matrix.Inverse();
            return new self.transform(Minv.Multiply(trnsf.traslation.Subtract(this.traslation)),Minv.Multiply(trnsf.matrix));
        }
        this.InverseTransform = function() {
            var Minv = this.matrix.Inverse();
            return new self.transform(Minv.Multiply(this.traslation).Oposite(),Minv);
        }
        this.setTranslation = function(pos) {
            if (pos instanceof self.Vector3) {
                this.traslation = pos;
            } else {
                this.traslation = new self.Vector3(pos);
            }
        }
        this.setMatrix = function(mat) {
            if (mat instanceof self.Matrix3x3) {
                this.matrix = mat;
            } else {
                this.matrix.value = mat;
            }
        }
        this.rotateThisAroundXAxis = function(a) {
            this.matrix.rotateThisAroundXAxis(a)
        }
        this.rotateThisAroundYAxis = function(a) {
            this.matrix.rotateThisAroundYAxis(a)
        }
        this.rotateThisAroundZAxis = function(a) {
            this.matrix.rotateThisAroundZAxis(a)
        }
        this.Transpose = function() {
            return new self.transform(this.matrix.Multiply(this.traslation),this.matrix);
        }
        this.TransposeInverse = function() {
            return new self.transform(this.matrix.Inverse().Multiply(this.traslation),this.matrix);
        }
        this.ToMatrix3x4 = function() {
            var t = this.matrix.Multiply(this.traslation);
            return new self.Matrix3x4([
                [this.matrix.value[0][0],this.matrix.value[0][1],this.matrix.value[0][2],t.value[0]],
                [this.matrix.value[1][0],this.matrix.value[1][1],this.matrix.value[1][2],t.value[1]],
                [this.matrix.value[2][0],this.matrix.value[2][1],this.matrix.value[2][2],t.value[2]],
            ]);
        }
        this.ToMatrixTransposed3x4 = function() {
            return new self.Matrix3x4([
                [this.matrix.value[0][0],this.matrix.value[0][1],this.matrix.value[0][2],this.traslation.value[0]],
                [this.matrix.value[1][0],this.matrix.value[1][1],this.matrix.value[1][2],this.traslation.value[1]],
                [this.matrix.value[2][0],this.matrix.value[2][1],this.matrix.value[2][2],this.traslation.value[2]],
            ]);
        }
        this.Move = function(pos) {
            if (pos instanceof self.Vector3) {
                this.traslation.AddToThis(pos);
            } else {
                this.traslation.AddToThis(new self.Vector3(pos));
            }
        }
        this.ScaleThis = function(scale) {
            this.matrix.ScaleThisBy(scale);
        }
        this.SetScale = function(scale) {
            this.matrix.SetScale(scale);
        }
		this.__proto__.toArrayBuffer = function() {
			var arr = [];
			arr.push(self.TextEncoder.encode("Transform\x01"));
			var F2A = new Float32Array([this.matrix.value[0][0],this.matrix.value[0][1],this.matrix.value[0][2],this.matrix.value[1][0],this.matrix.value[1][1],this.matrix.value[1][2],this.matrix.value[2][0],this.matrix.value[2][1],this.matrix.value[2][2],this.traslation.value[0],this.traslation.value[1],this.traslation.value[2]]);
			arr.push(new Uint32Array([F2A.buffer.byteLength,0]));
			arr.push(F2A);
			return self.State.concatArrayOfBuffers(arr);
		}
    }
    this.Object = function() {
        this.transform = new self.transform();
        this.Components = [];
        var ThisObject = this;
        this.Name = "New Empty Object";
        this.AddComponent = function(Component) {
            Component.Parrent = ThisObject;
            this.Components.push(Component);
        }
        this.Children = [];
        this.addChild = function(Child) {
            Child.Parrent = ThisObject;
            this.Children.push(Child);
        }
		this.Duplicate = function() {
			var newObj = new self.Object();
			newObj = this.transform.Clone();
			for (var i=0; i<this.Children.length; i++) {
				newObj.addChild(this.Children[i].Clone());
			}
			return newObj;
		}
		this.addChildKeepTransform = function(Child) {
			Child.transform = this.getGlobalTransform().InverseTransformTransform(Child.transform);
            Child.Parrent = ThisObject;
            this.Children.push(Child);
        }
		this.findChildByName = function(nme,dpth) {
			for (var i=0; i<this.Children.length; i++) {
				if (this.Children[i].Name == nme) {
					return this.Children[i];
				} else {
					if (dpth > 0) {
						var tmp = this.Children[i].findChildByName(nme,dpth-1);
						if (tmp) {
							return tmp;
						}
					}
				}
			}
			return false;
		}
        this.getGlobalTransform = function() {
            if (this.Parrent) {
                return this.Parrent.getGlobalTransform().TransformTransform(this.transform);
                // return this.transform.TransformTransform(this.Parrent.getGlobalTransform());
            } else {
                return this.transform;
            }
        }
        this.rotateThisAroundXAxis = function(a) {
            this.transform.rotateThisAroundXAxis(a)
        }
        this.rotateThisAroundThisXAxis = function(a) {
            this.transform.matrix.rotateThisAroundThisXAxis(a)
        }
		this.rotateJustThisAroundThisXAxis = function(a) {
			var m = new self.Matrix3x3(this.transform.matrix.value);
            this.rotateThisAroundThisXAxis(a);
			m = m.Multiply(this.transform.matrix.Inverse());
            this.MultiplyThisMatrixBy(m.Inverse());
			for (var i=0; i<this.Children.length; i++) {
				this.Children[i].MultiplyThisMatrixBy(m);
				this.Children[i].SetPosition(m.Multiply(this.Children[i].transform.traslation));
			}
        }
		this.rotateJustThisAroundXAxis = function(a) {
			var m = new self.Matrix3x3(this.transform.matrix.value);
            this.rotateThisAroundXAxis(a);
			m = m.Multiply(this.transform.matrix.Inverse());
            this.MultiplyThisMatrixBy(m.Inverse());
			for (var i=0; i<this.Children.length; i++) {
				this.Children[i].MultiplyThisMatrixBy(m);
				this.Children[i].SetPosition(m.Multiply(this.Children[i].transform.traslation));
			}
        }
        this.rotateThisAroundYAxis = function(a) {
            this.transform.rotateThisAroundYAxis(a)
        }
        this.rotateThisAroundThisYAxis = function(a) {
            this.transform.matrix.rotateThisAroundThisYAxis(a)
        }
		this.rotateJustThisAroundThisYAxis = function(a) {
			var m = new self.Matrix3x3(this.transform.matrix.value);
            this.rotateThisAroundThisYAxis(a);
			m = m.Multiply(this.transform.matrix.Inverse());
            this.MultiplyThisMatrixBy(m.Inverse());
			for (var i=0; i<this.Children.length; i++) {
				this.Children[i].MultiplyThisMatrixBy(m);
				this.Children[i].SetPosition(m.Multiply(this.Children[i].transform.traslation));
			}
        }
		this.rotateJustThisAroundYAxis = function(a) {
			var m = new self.Matrix3x3(this.transform.matrix.value);
            this.rotateThisAroundYAxis(a);
            m = m.Multiply(this.transform.matrix.Inverse());
            this.MultiplyThisMatrixBy(m.Inverse());
            for (var i=0; i<this.Children.length; i++) {
                this.Children[i].MultiplyThisMatrixBy(m);
                this.Children[i].SetPosition(m.Multiply(this.Children[i].transform.traslation));
            }
        }
        this.rotateThisAroundZAxis = function(a) {
            this.transform.rotateThisAroundZAxis(a);
        }
        this.rotateThisAroundThisZAxis = function(a) {
            this.transform.matrix.rotateThisAroundThisZAxis(a);
        }
		this.rotateJustThisAroundThisZAxis = function(a) {
			var m = new self.Matrix3x3(this.transform.matrix.value);
            this.rotateThisAroundThisZAxis(a);
            m = m.Multiply(this.transform.matrix.Inverse());
            this.MultiplyThisMatrixBy(m.Inverse());
            for (var i=0; i<this.Children.length; i++) {
                this.Children[i].MultiplyThisMatrixBy(m);
                this.Children[i].SetPosition(m.Multiply(this.Children[i].transform.traslation));
            }
        }
		this.rotateJustThisAroundZAxis = function(a) {
            var m = new self.Matrix3x3(this.transform.matrix.value);
            this.rotateThisAroundZAxis(a);
            m = m.Multiply(this.transform.matrix.Inverse());
            this.MultiplyThisMatrixBy(m.Inverse());
            for (var i=0; i<this.Children.length; i++) {
                this.Children[i].MultiplyThisMatrixBy(m);
                this.Children[i].SetPosition(m.Multiply(this.Children[i].transform.traslation));
            }
        }
        this.ScaleThis = function(s) {
            this.transform.ScaleThis(s);
        }
        this.SetTransform = function(trnsf) {
            this.transform = trnsf;
        }
        this.SetScale = function(s) {
            this.transform.SetScale(s);
        }
        this.SetPosition = function(pos) {
            this.transform.setTranslation(pos);
        }
		this.MovePosition = function(pos) {
            this.transform.Move(pos);
        }
        this.setMatrix = function(mat) {
            this.transform.setMatrix(mat);
        }
		this.MultiplyThisMatrixBy = function(mt) {
			this.transform.matrix.MultiplyThisBy(mt);
		}
		this.__proto__.toArrayBuffer = function() {
			var arr = [];
			arr.push(self.TextEncoder.encode("Object\x00"));
			arr.push(new Uint32Array([0,0]));
			var Name = self.TextEncoder.encode(this.Name);
			arr.push(self.TextEncoder.encode("Name\x01"));
			arr.push(new Uint32Array([Name.length,2]));
			arr.push(Name);
			arr.push(this.transform.toArrayBuffer());
			arr.push(self.TextEncoder.encode("Children\x00"));
			arr.push(new Uint32Array([0,0]));
			var Length0 = 0;
			for (var i=0; i<this.Children.length; i++) {
				var bf = this.Children[i].toArrayBuffer();
				arr.push(bf);
				Length0 += bf.length;
			}
			arr[7] = new Uint32Array([Length0,0]);
			arr.push(self.TextEncoder.encode("Components\x00"));
			var idx = arr.length;
			arr.push(new Uint32Array([0,0]));
			var Length2 = 0;
			for (var i=0; i<this.Components.length; i++) {
				var o = {};
				//var Name = Object.keys(self.Components)[Object.values(self.Components).indexOf(this.Components[i].constructor)];
				var Name = self.State.getComponentType(this.Components[i]);
				o[Name] = this.Components[i];
				var bf = self.State.objectToArrayBuffer(o);
				arr.push(bf);
				Length2 += bf.length;
			}
			arr[idx] = new Uint32Array([Length2,0]);
			var Length1 = 0;
			for (var i=2; i<arr.length; i++) {
				Length1 += arr[i].buffer.byteLength;
			}
			arr[1] = new Uint32Array([Length1,0]);
			return self.State.concatArrayOfBuffers(arr);
		}
    }
	this.Math = {};
	this.Math.ComplexNumber = function(a,b) {
		this.real = a;
		this.imaginary = b;
		this.Add = function(c) {
			return new self.Math.ComplexNumber(this.real+c.real,this.imaginary+c.imaginary);
		}
		this.AddThis = function(c) {
			this.real += c.real;
			this.imaginary += c.imaginary;
			return this;
		}
		this.Subtract = function(c) {
			return new self.Math.ComplexNumber(this.real-c.real,this.imaginary-c.imaginary);
		}
		this.Oposite = function() {
			return new self.Math.ComplexNumber(-this.real,-this.imaginary);
		}
		this.Inverse = function() {
			var l = (this.real*this.real)+(this.imaginary*this.imaginary);
			return new self.Math.ComplexNumber(this.real/l,-this.imaginary/l);
		}
		this.Multiply = function(c) {
			if (c instanceof self.Math.ComplexNumber) {
				return new self.Math.ComplexNumber((this.real*c.real)-(this.imaginary*c.imaginary),(this.imaginary*c.real)+(this.real*c.imaginary));
			} else {
				return new self.Math.ComplexNumber(this.real*c,this.imaginary*c);
			}
		}
		this.Divide = function(c) {
			return this.Multiply(c.Inverse());
		}
		this.abs = function() {
			return Math.hypot(this.real,this.imaginary);
		}
		this.ln = function() {
			return new self.Math.ComplexNumber(Math.log(this.abs()),Math.atan2(this.imaginary,this.real));
		}
		this.exp = function() {
			var l = Math.exp(this.real);
			return new self.Math.ComplexNumber(l*Math.cos(this.imaginary),l*Math.sin(this.imaginary));
		}
		this.pow = function(c) {
			var l = this.ln();
			if (isNaN(l.real) || isNaN(l.imaginary)) {
				return new self.Math.ComplexNumber(0,0);
			} else {
				return l.Multiply(c).exp();
			}
		}
	}
	this.Math.One = new self.Math.ComplexNumber(1,0);
	this.Math.Two = new self.Math.ComplexNumber(2,0);
	this.Math.Pi = new self.Math.ComplexNumber(Math.PI,0);
	this.Math.PiOver2 = new self.Math.ComplexNumber(Math.PI/2,0);
	this.Math.sin = function(c) {
		var ep = Math.exp(c.imaginary)/2;
		var en = 0.25/ep;
		return new self.Math.ComplexNumber((ep+en)*Math.sin(c.real),(ep-en)*Math.cos(c.real));
	}
	this.Math.cos = function(c) {
		var ep = Math.exp(c.imaginary)/2;
		var en = 0.25/ep;
		return new self.Math.ComplexNumber((ep+en)*Math.cos(c.real),(en-ep)*Math.sin(c.real));
	}
	this.Math.gammaFunction = function(c) {
		if (c.real < 0) {
			return self.Math.sin(c.Multiply(Math.PI)).Multiply(self.Math.gammaFunction(new self.Math.ComplexNumber(1-c.real,-c.imaginary))).pow(new self.Math.ComplexNumber(-1,0)).Multiply(Math.PI);
		} else {
			c = new self.Math.ComplexNumber(c.real-1,c.imaginary);
			var prec = 0.1;
			var v = new self.Math.ComplexNumber(0,0);
			var curr = new self.Math.ComplexNumber(0,0);
			for (var x=new self.Math.ComplexNumber(prec,0); (curr.abs() > 0.01 || x.real < 2); x.real+=prec) {
				curr = x.pow(c).Multiply(Math.exp(-x.real));
				v = v.Add(curr.Multiply(prec));
			}
			return v;
		}
	}
	this.Math.RiemannZetaFunction = function(c) {
		if (c.real < 0.5) {
			return self.Math.Two.pow(c).Multiply(self.Math.Pi.pow(c.Subtract(self.Math.One))).Multiply(self.Math.sin(self.Math.PiOver2.Multiply(c))).Multiply(self.Math.gammaFunction(self.Math.One.Subtract(c))).Multiply(self.Math.RiemannZetaFunction(self.Math.One.Subtract(c)));
		} else {
			var result = new self.Math.ComplexNumber(0,0);
			c = c.Oposite();
			for (var i=new self.Math.ComplexNumber(1,0); i.real<256; i.real++) {
				result.AddThis(i.pow(c));
			}
			return result;
		}
	}
    this.IdentityTransform = new self.transform();
	this.IdentityQuaternion = new self.Quaternion([0,0,0,1]);
	this.IdentityMatrix = new self.Matrix3x3();
    this.Root = new this.Object();
    this.Root.Name = "Root";
	this.Assets = [];
	this.listAssets = function() {
		function addAsset(asslol) {
			if (!self.Assets.includes(asslol)) {
				self.Assets.push(asslol);
			}
		}
		function findAssets(oj) {
			for (var i=0; i<oj.Components.length; i++) {
				if (oj.Components[i] instanceof self.Components.MeshRenderer) {
					addAsset(oj.Components[i].Mesh);
					for (var j=0; j<oj.Components[i].Materials.length; j++) {
						addAsset(oj.Components[i].Materials[j]);
						addAsset(oj.Components[i].Materials[j].shader);
						var Vals = Object.keys(oj.Components[i].Materials[j].Values);
						for (var k=0; k<Vals.length; k++) {
							if (oj.Components[i].Materials[j].Values[Vals[k]] instanceof self.Texture) {
								addAsset(oj.Components[i].Materials[j].Values[Vals[k]]);
							}
						}
					}
				} else if (oj.Components[i] instanceof self.Components.SkinnedMeshRenderer) {
					addAsset(oj.Components[i].SkinnedMesh);
					for (var j=0; j<oj.Components[i].Materials.length; j++) {
						addAsset(oj.Components[i].Materials[j]);
						addAsset(oj.Components[i].Materials[j].shader);
						var Vals = Object.keys(oj.Components[i].Materials[j].Values);
						for (var k=0; k<Vals.length; k++) {
							if (oj.Components[i].Materials[j].Values[Vals[k]] instanceof self.Texture) {
								addAsset(oj.Components[i].Materials[j].Values[Vals[k]]);
							}
						}
					}
				}
			}
			for (var i=0; i<oj.Children.length; i++) {
				findAssets(oj.Children[i]);
			}
		}
		findAssets(self.Root);
		return self.Assets;
	}
	this.cleanUnusedAssets = function() {
		this.Assets = [];
		self.listAssets();
		self.gl.flush();
	}
    this.CreateMeshObject = function(mesh,Mats) {
        var MeshObject = new self.Object();
        MeshObject.Name = "New Mesh";
        var MeshComponent = new self.Components.MeshRenderer(mesh,Mats);
        MeshObject.AddComponent(MeshComponent);
        return MeshObject;
    }
    //this.DebugBoneMesh = new this.Mesh({VertexPositions:[[0.001,0.15,0.001],[-0.001,0.15,0.001],[0.01,-0.01,0.01],[-0.01,-0.01,0.01],[0.001,0.15,-0.001],[-0.001,0.15,-0.001],[0.01,-0.01,-0.01],[-0.01,-0.01,-0.01]],VertexNormals:[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],VertexDataIndexs:[[0,0,0],[2,0,0],[4,0,0],[6,0,0],[1,1,0],[3,1,0],[5,1,0],[7,1,0],[0,2,0],[1,2,0],[4,2,0],[5,2,0],[2,3,0],[3,3,0],[6,3,0],[7,3,0],[0,4,0],[1,4,0],[2,4,0],[3,4,0],[4,5,0],[5,5,0],[6,5,0],[7,5,0]],Indices:[[[0,1,2],[3,2,1],[4,6,5],[7,5,6],[8,10,9],[11,9,10],[12,13,14],[15,14,13],[16,17,18],[19,18,17],[20,22,21],[23,21,22]]]});
    this.CreateSkinnedMeshObject = function(SkinnedMesh,Mats) {
        var SkinnedMeshObject = new self.Object();
        SkinnedMeshObject.Name = "New Skinned Mesh";
        var SkinnedMeshComponent = new self.Components.SkinnedMeshRenderer(SkinnedMesh,Mats);
        SkinnedMeshObject.AddComponent(SkinnedMeshComponent);
        for (var i = 0; i < SkinnedMesh.Bones.length; i++) {
            var Bone = new self.Object();
            Bone.Name = SkinnedMesh.Bones[i];
            Bone.transform = SkinnedMeshComponent.DefaultPose[i];
            // var BoneRef = new self.Object();
            // BoneRef.Name = "BoneRef";
            // BoneRef.transform = MeshComponent.DefaultPose[i].InverseTransformTransform(self.IdentityTransform);
            // MeshComponent.SetBoneRefrence(i,BoneRef);
            // Bone.addChild(BoneRef);
            // var DebugBoneMeshComponent = new self.Components.MeshRenderer(self.DebugBoneMesh,Mats);
            // Bone.AddComponent(DebugBoneMeshComponent);
            SkinnedMeshComponent.SetBoneRefrence(i,Bone);
            // SkinnedMeshObject.addChild(Bone);
        }
        return SkinnedMeshObject;
    }
    this.CreateCamera = function(canv,res,Fov,Clip,VR,IPD) {
        var CameraObject = new self.Object();
        CameraObject.Name = "New Camera";
        var CameraComponent = new self.Components.CamreaViewPort(canv,res,Fov,Clip,VR,IPD);
        CameraObject.AddComponent(CameraComponent);
        return CameraObject;
    }
	this.Update = function(obj) {
		obj = obj || self.Root;
		for (var i=0; i<obj.Components.length; i++) {
			if (obj.Components[i].Update instanceof Function) {
				obj.Components[i].Update();
			}
		}
		for (var i=0; i<obj.Children.length; i++) {
			self.Update(obj.Children[i]);
		}
	}
	this.UpdateRenderQue = function() {
		self.RenderQue = [];
		self.Lights = [];
		self.ReflectionProbes = [];
		function renderChildren(Obj) {
			for (var i = 0; i < Obj.Children.length; i++) {
				for (var j=0; j<Obj.Children[i].Components.length; j++) {
					if ((Obj.Children[i].Components[j] instanceof self.Components.MeshRenderer) || (Obj.Children[i].Components[j] instanceof self.Components.SkinnedMeshRenderer)) {
						self.RenderQue.push(Obj.Children[i].Components[j]);
					}
					if ((Obj.Children[i].Components[j] instanceof self.Components.PointLight) || (Obj.Children[i].Components[j] instanceof self.Components.DirectionalLight)) {
						var data = Obj.Children[i].Components[j].getData();
						if (data) {
							self.Lights.push(data);
						}
					}
					if (Obj.Children[i].Components[j] instanceof self.Components.ReflectionProbe) {
						self.ReflectionProbes.push(Obj.Children[i].Components[j].getData());
					}
				}
				renderChildren(Obj.Children[i]);
			}
		}
		renderChildren(self.Root);
	}
    this.Render = function() {
		self.UpdateRenderQue();
        function RenderCameras(obj) {
            for (var i = 0; i < obj.Components.length; i++) {
                if (obj.Components[i] instanceof self.Components.CamreaViewPort) {
                    obj.Components[i].renderFrame();
                }
            }
            for (var i = 0; i < obj.Children.length; i++) {
                RenderCameras(obj.Children[i]);
            }
        }
        RenderCameras(this.Root);
    }
	this.State = {};
	this.State.concatArrayOfBuffers = function(arr) {
		var length = 0;
		for (var i=0; i<arr.length; i++) {
			length += arr[i].buffer.byteLength;
		}
		var bufr = new Uint8Array(length);
		var idx = 0;
		for (var i=0; i<arr.length; i++) {
			bufr.set(new Uint8Array(arr[i].buffer),idx);
			idx += arr[i].buffer.byteLength;
		}
		return bufr;
	}
	this.State.toArrayBuffer = function(thing) {
		var arr = [];
		self.cleanUnusedAssets();
		for (var i=0; i<self.Assets.length; i++) {
			if (self.Assets[i].toArrayBuffer) {
				arr.push(self.Assets[i].toArrayBuffer());
			}
		}
		arr.push(self.Root.toArrayBuffer());
		//return self.State.concatArrayOfBuffers(arr);
		
		//var AssetsBuffer = self.State.concatArrayOfBuffers(arr);
		//arr = [self.TextEncoder.encode("Assets\x00"),new Uint32Array([AssetsBuffer.length,0]),AssetsBuffer];
		//return self.State.concatArrayOfBuffers(arr);
		
		return self.State.concatArrayOfBuffers(arr);
	}
	this.State.parseArrayBuffer = function(ab,name) {
		var idx = 0;
		var Length = 0;
		var state = 0;
		var Name = "";
		var isData = false;
		var Type = 0;
		var Result = name ? [name] : [];
		for (var i=0; i<ab.length;) {
			if (state === 0) {
				if (ab[i] === 0 || ab[i] === 1) {
					state++;
					isData = (ab[i] === 1);
				} else {
					i++;
				}
			} else if (state === 1) {
				Name = self.TextDecoder.decode(ab.subarray(idx,i));
				i++;
				idx = i;
				state++;
			} else if (state === 2) {
				Length = new Uint32Array(ab.slice(i,i+8).buffer);
				Type = Length[1];
				Length = Length[0];
				state++;
				i += 8;
			} else {
				if (isData) {
					var Data = ab.slice(i,i+Length);
					i+=Length;
					idx = i;
					state = 0;
					if (Type === 0) {
						Data = new Float32Array(Data.buffer);
					} else if (Type === 1) {
						Data = new Int32Array(Data.buffer);
					} else if (Type === 2) {
						Data = self.TextDecoder.decode(Data);
					} else if (Type === 4) {
						Data = new Float32Array(pako.ungzip(Data).buffer);
					} else if (Type === 5) {
						Data = new Int32Array(pako.ungzip(Data).buffer);
					} else if (Type === 6) {
						Data = pako.ungzip(Data)
					} else if (Type === 7) {
						Data = new Uint16Array(Data.buffer);
					}
					Result.push([Name,Data]);
				} else {
					Result.push(self.State.parseArrayBuffer(ab.slice(i,i+Length),Name));
					i+=Length;
					idx = i;
					state = 0;
				}
			}
		}
		return Result;
	}
	this.State.objectToArrayBuffer = function(obj) {
		var arr = [];
		var keys = Object.keys(obj);
		for (var i=0; i<keys.length; i++) {
			var val = obj[keys[i]];
			if (val) {
				if (val instanceof Object && val.constructor !== Array && val.constructor !== self.Object && val.constructor !== Function) {
					var bfr = self.State.objectToArrayBuffer(val);
					if (bfr.length > 0) {
						arr.push(self.TextEncoder.encode(keys[i]+"\x00"));
						arr.push(new Uint32Array([bfr.buffer.byteLength,0]));
						arr.push(bfr);
					}
				} else if (val.constructor === Array) {
					var isNumbers = true;
					for (var j=0; j<val.length; j++) {
						if (val[j].constructor !== Number) {
							isNumbers = false;
							break;
						}
					}
					if (isNumbers) {
						arr.push(self.TextEncoder.encode(keys[i]+"\x01"));
						var bfr = new Float32Array(val);
						arr.push(new Uint32Array([bfr.buffer.byteLength,0]));
						arr.push(bfr);
					} else {
						for (var j=0; j<val.length; j++) {
							if (val[j].constructor === Number) {
								arr.push(self.TextEncoder.encode(keys[i]+"["+j.toString()+"]\x01"));
								var bfr = new Float32Array([val]);
								arr.push(new Uint32Array([bfr.buffer.byteLength,0]));
								arr.push(bfr);
							} else {
								var bfr = self.State.objectToArrayBuffer(val);
								if (bfr.length > 0) {
									arr.push(self.TextEncoder.encode(keys[i]+"["+j.toString()+"]\x00"));
									arr.push(new Uint32Array([bfr.buffer.byteLength,0]));
									arr.push(bfr);
								}
							}
						}
					}
				} else if (val instanceof Float32Array) {
					arr.push(self.TextEncoder.encode(keys[i]+"\x01"));
					arr.push(new Uint32Array([val.buffer.byteLength,0]));
					arr.push(bfr);
				} else if (val.constructor === String) {
					arr.push(self.TextEncoder.encode(keys[i]+"\x01"));
					var bfr = self.TextEncoder.encode(val);
					arr.push(new Uint32Array([bfr.buffer.byteLength,2]));
					arr.push(bfr);
				}
			}
		}
		if (arr.length > 1) {
			return self.State.concatArrayOfBuffers(arr);
		} else {
			return new Uint8Array(0);
		}
	}
	this.State.getComponentType = function(thing,depth,search) {
		if (depth === undefined) {
			depth = 4;
		}
		if (depth <= 0) {
			return undefined;
		}
		search = search || self.Components;
		var keys = Object.keys(search);
		for (var i=0; i<keys.length; i++) {
			if (thing.constructor === search[keys[i]]) {
				return keys[i];
			} else {
				var s = self.State.getComponentType(thing,depth,search[keys[i]]);
				if (s) {
					return keys[i]+"."+s;
				}
			}
		}
	}
	this.State.load = function(arrbfr) {
		var data = self.State.parseArrayBuffer(arrbfr);
		var Assets = [];
		var Root = null;
		function toObject(arr) {
			var Type = arr.splice(0,1)[0];
			var obj = new self.Object();
			for (var i=0; i<arr.length; i++) {
				if (arr[i][0] == "Name") {
					obj.Name = arr[i][1];
				} else if (arr[i][0] == "Transform") {
					//obj.Name = arr[i][1];
					var val = arr[i][1];
					var M = new self.Matrix3x3([[val[0],val[1],val[2]],[val[3],val[4],val[5]],[val[6],val[7],val[8]]]);
					var V = new self.Vector3([val[9],val[10],val[11]]);
					
				} else if (arr[i][0] == "Children") {
					for (var j=1; j<arr[i].length; j++) {
						obj.addChild(toObject(arr[i][j]));
					}
				}
			}
			return obj;
		}
		for (var i=0; i<data.length; i++) {
			if (data[i][0] == "Object") {
				Root = toObject(data[i]);
			} else {
				Assets.push(data[i]);
			}
		}
		return {Root,Assets};
	}
}