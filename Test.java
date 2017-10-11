import java.io.*;
import java.util.*;

static class Test{
	public static void main(String [] args){
		List<Integer> li = new ArrayList<Integer>();
        li.add(1);
        li.add(2);
	Iterator<Integer> it = li.iterator();
	while(it.hasNext()){
		//li.add(3);
		System.out.println(it.next());
	}
	}
}
