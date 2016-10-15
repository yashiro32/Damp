package en.menghui.android.damp.recurrent.lstm;

import java.util.ArrayList;

import Jama.Matrix;

public class LSTMNetwork {
	public LSTMParam lstmParam;
	public ArrayList<LSTMNode> lstmNodeList;
	public ArrayList<Matrix> xList;
	
	public LSTMNetwork(LSTMParam lstmParam) {
		this.lstmParam = lstmParam;
		this.lstmNodeList = new ArrayList<LSTMNode>();
		// Input sequence.
		this.xList = new ArrayList<Matrix>();
	}
	
	public void yListIs(double[] yList) {
		int idx = this.xList.size() - 1;
		
		Matrix diffH = LossLayer.bottomDiff(lstmNodeList.get(idx).state.h, yList[idx]);
		
		Matrix diffS = new Matrix (this.lstmParam.memCellCt, 1, 0.0);
		this.lstmNodeList.get(idx).backProp(diffH, diffS);
		idx--;
		
		while(idx >= 0) {
			diffH = LossLayer.bottomDiff(this.lstmNodeList.get(idx).state.h, yList[idx]);
			diffH.plusEquals(this.lstmNodeList.get(idx+1).state.bdh);
			diffS = this.lstmNodeList.get(idx+1).state.bds;
			this.lstmNodeList.get(idx).backProp(diffH, diffS);
			idx--;
		}
	}
	
	public void xListClear() {
		this.xList = new ArrayList<Matrix>();
	}
	
	public void xListAdd(Matrix x) {
		this.xList.add(x);
		if (this.xList.size() > this.lstmNodeList.size()) {
			LSTMState lstmState = new LSTMState(this.lstmParam.memCellCt, this.lstmParam.xDim);
			this.lstmNodeList.add(new LSTMNode(this.lstmParam, lstmState));
		}
		
		// Get index of most recent x input
		int idx = this.xList.size() - 1;
		if (idx == 0) {
			// No recurrent inputs yet
			this.lstmNodeList.get(idx).forwardProp(x, null, null);
		} else {
			Matrix sPrev = this.lstmNodeList.get(idx-1).state.s;
			Matrix hPrev = this.lstmNodeList.get(idx-1).state.h;
			this.lstmNodeList.get(idx).forwardProp(x, sPrev, hPrev);
		}
	}
	
	
}
