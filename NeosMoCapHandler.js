var NeosMotion = function() {
	this.MotionData = [];
	this.Header = [];
	this.fromArrayBuffer = function(data) {
		this.MotionData = [];
		this.Header = [];
		var headerLen = data[0];
		var decoder = new TextDecoder();
		var idx0 = 1;
		var idx1 = 1;
		for (var i=0; i<headerLen;) {
			if (data[idx0] === 0) {
				this.Header.push({Name:decoder.decode(data.subarray(idx1,idx0)),Type:data[idx0+1],Bytes:data[idx0+2]});
				idx0+=2;
				idx1 = idx0+1;
				i++;
			} else {
				idx0++;
			}
		}
		data = data.slice(idx0+1);
		for (var i=0; i<data.length;) {
			var obj = {};
			for (var j=0; j<this.Header.length; j++) {
				if (this.Header[j].Type === 0) {
					obj[this.Header[j].Name] = new Float32Array(data.slice(i,i+this.Header[j].Bytes).buffer);
					i+=this.Header[j].Bytes;
				} else if (this.Header[j].Type === 1) {
					obj[this.Header[j].Name] = !!data[i];
					i++;
				} else if (this.Header[j].Type === 2) {
					obj[this.Header[j].Name] = new Date(Number(new BigUint64Array(data.slice(i,i+8).buffer)[0]));
					i+=8;
				}
			}
			this.MotionData.push(obj);
		}
	}
	this.findRawPoseAtTimeInMs = function(time) {
		time += this.MotionData[0].Time.valueOf();
		var idx = 0;
		for (var i=Math.floor(Math.log2(this.MotionData.length)); i>=0; i--) {
			idx = Math.max(Math.min(idx,this.MotionData.length-1),0);
			if (this.MotionData[idx].Time > time) {
				idx -= 2**i;
			} else {
				idx += 2**i;
			}
		}
		idx = Math.max(Math.min(Math.round(idx),this.MotionData.length-1),0);
		return this.MotionData[idx];
	}
	this.interpolate = function(v0,v1,t) {
		if (v0.constructor === Float32Array || v0.constructor === Array) {
			if (v0.length === 1) {
				return new Float32Array([((1-t)*v0[0])+(t*v1[0])]);
			} else if (v0.length === 2) {
				return new Float32Array([((1-t)*v0[0])+(t*v1[0]),((1-t)*v0[1])+(t*v1[1])]);
			} else if (v0.length === 3) {
				return new Float32Array([((1-t)*v0[0])+(t*v1[0]),((1-t)*v0[1])+(t*v1[1]),((1-t)*v0[2])+(t*v1[2])]);
			} else if (v0.length === 4) {
				var d = (v0[0]*v1[0])+(v0[1]*v1[1])+(v0[2]*v1[2])+(v0[3]*v1[3]);
				if (d < 0) {
					v1[0] = -v1[0];
					v1[1] = -v1[1];
					v1[2] = -v1[2];
					v1[3] = -v1[3];
					d = -d;
				}
				if (d > 0.95) {
					var v = new Float32Array([((1-t)*v0[0])+(t*v1[0]),((1-t)*v0[1])+(t*v1[1]),((1-t)*v0[2])+(t*v1[2]),((1-t)*v0[3])+(t*v1[3])]);
					var l = Math.hypot(v[0],v[1],v[2],v[3]);
					v[0] /= l;
					v[1] /= l;
					v[2] /= l;
					v[3] /= l;
					return v;
				} else {
					var theta = Math.acos(d);
					var div = Math.sin(theta);
					var a = Math.sin((1-t)*theta)/div;
					var b = Math.sin(t*theta)/div;
					return new Float32Array([(a*v0[0])+(b*v1[0]),(a*v0[1])+(b*v1[1]),(a*v0[2])+(b*v1[2]),(a*v0[3])+(b*v1[3])]);
				}
			}
		} else {
			return v0;
		}
	}
	this.findPoseAtTimeInMs = function(time) {
		time += this.MotionData[0].Time.valueOf();
		var idx = 0;
		for (var i=Math.floor(Math.log2(this.MotionData.length)); i>=0; i--) {
			idx = Math.max(Math.min(Math.round(idx),this.MotionData.length-1),0);
			if (this.MotionData[idx].Time > time) {
				idx -= 2**i;
			} else {
				idx += 2**i;
			}
		}
		idx = Math.max(Math.min(Math.round(idx),this.MotionData.length-1),0);
		var p2 = this.MotionData[idx];
		if (p2.Time > time) {
			var p1 = p2;
			var p0 = this.MotionData[Math.max(idx-1,0)];
		} else {
			var p1 = this.MotionData[Math.min(idx+1,this.MotionData.length-1)];
			var p0 = p2;
		}
		if (p0 === p1) {
			return p0;
		}
		var t = (time-p0.Time)/(p1.Time-p0.Time);
		var p = {};
		for (var i=0; i<this.Header.length; i++) {
			p[this.Header[i].Name] = this.interpolate(p0[this.Header[i].Name],p1[this.Header[i].Name],t);
		}
		return p;
	}
}