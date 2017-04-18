package common.processor;

public interface Processor<T>
{
	public void process(T t);
	public void output();
}
