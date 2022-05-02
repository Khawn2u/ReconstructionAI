var GraphHandler = function(canvas) {
    this.canvas = canvas;
	this.ctx = this.canvas.getContext("2d", {alpha: true});
	this.ctx.fillStyle = "#000000";
	this.ctx.fillRect(0,0,this.canvas.width,this.canvas.height);
	this.avgGraph = [0,0.3333333333333333,0.3333333333333333,0.3333333333333333,0];
	this.Graph = [];
	this.SmoothGrapth = [];
	this.SmoothGrapthTmp = [];
	this.SmoothGrapthSum = [];
	this.calculateSmoothGrapthLookup = function(s) {
		var avgGraph = new Float64Array((s*2)+1).fill(0);
		avgGraph[(avgGraph.length-1)/2] = 1;
		for (var i=0; i<(avgGraph.length-3)/2; i++) {
			avgGraph = avgGraph.map(function(xin,idx){
				 if (idx > 0) {
					  if (idx < avgGraph.length-1) {
							return (xin+avgGraph[idx-1]+avgGraph[idx+1])/3;
					  } else {
							return (xin+avgGraph[idx-1])/2;
					  }
				 } else {
					  return (xin+avgGraph[idx+1])/2;
				 }
			});
		}
		this.avgGraph = avgGraph;
	}
	this.addValue = function(v) {
		if (v || v == 0) {
			if (this.SmoothGrapthTmp.length < (this.avgGraph.length-2)) {
				this.SmoothGrapthTmp = new Array(this.avgGraph.length-2).fill(0);
				this.SmoothGrapthSum = new Array(this.avgGraph.length-2).fill(0);
			}
			this.Graph.push(v);
			for (var i=1; i<(this.avgGraph.length-1); i++) {
				var idx = Math.max(i+this.Graph.length-2,0);
				this.SmoothGrapthTmp[idx] += this.avgGraph[i]*v;
				this.SmoothGrapthSum[idx] += this.avgGraph[i];
			}
			this.SmoothGrapthTmp.push(0);
			this.SmoothGrapthSum.push(0);
			this.SmoothGrapth.push(0);
			for (var i=1; i<((this.avgGraph.length+1)/2); i++) {
				var avgtmpIdx = this.SmoothGrapthTmp.length-(i+((this.avgGraph.length+1)/2));
				var avgIdx = this.SmoothGrapth.length-i;
				this.SmoothGrapth[this.SmoothGrapth.length-i] = this.SmoothGrapthTmp[avgtmpIdx]/this.SmoothGrapthSum[avgtmpIdx];
			}
		}
	}
	this.GraphValues = function() {
		this.ctx.fillStyle = "#000000";
		this.ctx.fillRect(0,0,this.canvas.width,this.canvas.height);
		this.ctx.lineWidth = 3;
		this.ctx.strokeStyle = "#FF0000";
		this.ctx.beginPath();
		for (var x = 0; x < this.Graph.length; x++) {
			this.ctx.lineTo(x*(this.canvas.width/(this.Graph.length-1)),this.canvas.height-this.Graph[x]*this.canvas.height);
			this.ctx.moveTo(x*(this.canvas.width/(this.Graph.length-1)),this.canvas.height-this.Graph[x]*this.canvas.height);
		}
		this.ctx.stroke();
		this.ctx.lineWidth = 2;
		this.ctx.strokeStyle = "#00FF00";
		this.ctx.beginPath();
		for (var x = 0; x < this.SmoothGrapth.length; x++) {
			this.ctx.lineTo(x*(this.canvas.width/(this.SmoothGrapth.length-1)),this.canvas.height-this.SmoothGrapth[x]*this.canvas.height);
			this.ctx.moveTo(x*(this.canvas.width/(this.SmoothGrapth.length-1)),this.canvas.height-this.SmoothGrapth[x]*this.canvas.height);
		}
		this.ctx.stroke();
	}
	this.addValueAndGrapth = function(v) {
		this.addValue(v);
		this.GraphValues();
	}
	this.Clear = function() {
		this.ctx.fillStyle = "#000000";
		this.ctx.fillRect(0,0,this.canvas.width,this.canvas.height);
		this.Graph = [];
		this.SmoothGrapth = [];
		this.SmoothGrapthTmp = [];
		this.SmoothGrapthSum = [];
	}
}