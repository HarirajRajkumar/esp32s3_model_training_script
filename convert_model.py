# save this as convert_model.py
import onnx
import numpy as np
import struct
import os
import argparse

def convert_onnx_to_espdl(onnx_path, output_path, quantize=True):
    """
    Convert an ONNX model to ESP-DL format
    
    Args:
        onnx_path: Path to the input ONNX model
        output_path: Path to save the ESP-DL model
        quantize: Whether to quantize the model to int8
    """
    print(f"Loading ONNX model from {onnx_path}")
    model = onnx.load(onnx_path)
    
    # Verify the model
    try:
        onnx.checker.check_model(model)
        print("ONNX model is valid!")
    except Exception as e:
        print(f"ONNX model verification failed: {e}")
        return False
    
    # Extract model information
    print("Extracting model information...")
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    
    # Get input shape
    input_shape = []
    for dim in model.graph.input[0].type.tensor_type.shape.dim:
        if dim.dim_value:
            input_shape.append(dim.dim_value)
        else:
            # For dynamic dimensions, use a default
            input_shape.append(1)
    
    print(f"Input shape: {input_shape}")
    
    # Start creating ESP-DL model file
    with open(output_path, 'wb') as f:
        # Write magic header "ESPDL" + version (1.0)
        f.write(b'ESPDL\x01\x00')
        
        # Write input shape (N, C, H, W)
        for dim in input_shape:
            f.write(struct.pack('<I', dim))
        
        # Write model type (1 = classification)
        f.write(struct.pack('<I', 1))
        
        # Write quantization flag
        f.write(struct.pack('<?', quantize))
        
        # Extract weights and biases
        print("Extracting weights and biases...")
        
        # Write number of layers
        num_layers = len(model.graph.node)
        f.write(struct.pack('<I', num_layers))
        
        # Process each layer
        for i, node in enumerate(model.graph.node):
            print(f"Processing node {i+1}/{num_layers}: {node.op_type}")
            
            # Write operator type
            op_type_map = {
                'Conv': 1,
                'MaxPool': 2,
                'AveragePool': 3,
                'Relu': 4,
                'Gemm': 5,
                'Flatten': 6,
                'Softmax': 7,
                # Add more mappings as needed
            }
            
            op_code = op_type_map.get(node.op_type, 0)
            f.write(struct.pack('<I', op_code))
            
            # Write node inputs and outputs
            f.write(struct.pack('<I', len(node.input)))
            for inp in node.input:
                inp_bytes = inp.encode('utf-8')
                f.write(struct.pack('<I', len(inp_bytes)))
                f.write(inp_bytes)
                
            f.write(struct.pack('<I', len(node.output)))
            for out in node.output:
                out_bytes = out.encode('utf-8')
                f.write(struct.pack('<I', len(out_bytes)))
                f.write(out_bytes)
            
            # Write attributes
            f.write(struct.pack('<I', len(node.attribute)))
            for attr in node.attribute:
                attr_name = attr.name.encode('utf-8')
                f.write(struct.pack('<I', len(attr_name)))
                f.write(attr_name)
                
                # Write attribute type and value
                if attr.type == onnx.AttributeProto.INT:
                    f.write(struct.pack('<I', 1))  # INT type
                    f.write(struct.pack('<q', attr.i))
                elif attr.type == onnx.AttributeProto.FLOAT:
                    f.write(struct.pack('<I', 2))  # FLOAT type
                    f.write(struct.pack('<f', attr.f))
                elif attr.type == onnx.AttributeProto.STRING:
                    f.write(struct.pack('<I', 3))  # STRING type
                    s_bytes = attr.s
                    f.write(struct.pack('<I', len(s_bytes)))
                    f.write(s_bytes)
                elif attr.type == onnx.AttributeProto.INTS:
                    f.write(struct.pack('<I', 4))  # INTS type
                    f.write(struct.pack('<I', len(attr.ints)))
                    for val in attr.ints:
                        f.write(struct.pack('<q', val))
                elif attr.type == onnx.AttributeProto.FLOATS:
                    f.write(struct.pack('<I', 5))  # FLOATS type
                    f.write(struct.pack('<I', len(attr.floats)))
                    for val in attr.floats:
                        f.write(struct.pack('<f', val))
        
        # Extract initializers (weights and biases)
        print("Writing weights and biases...")
        f.write(struct.pack('<I', len(model.graph.initializer)))
        
        for tensor in model.graph.initializer:
            tensor_name = tensor.name.encode('utf-8')
            f.write(struct.pack('<I', len(tensor_name)))
            f.write(tensor_name)
            
            # Write shape
            shape = [dim for dim in tensor.dims]
            f.write(struct.pack('<I', len(shape)))
            for dim in shape:
                f.write(struct.pack('<I', dim))
            
            # Extract data
            np_array = onnx.numpy_helper.to_array(tensor)
            
            # Quantize if needed
            if quantize:
                # Scale to int8 range
                scale = 127.0 / max(abs(np_array.max()), abs(np_array.min())) if np_array.size > 0 else 1.0
                np_array = np.clip(np.round(np_array * scale), -128, 127).astype(np.int8)
                
                # Write scale factor
                f.write(struct.pack('<f', scale))
                
                # Write data type (1 = int8)
                f.write(struct.pack('<I', 1))
            else:
                # Write data type (0 = float32)
                f.write(struct.pack('<I', 0))
            
            # Write data
            np_array.tofile(f)
    
    print(f"Conversion complete! ESP-DL model saved to {output_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ONNX model to ESP-DL format')
    parser.add_argument('--input', type=str, required=True, help='Input ONNX model path')
    parser.add_argument('--output', type=str, required=True, help='Output ESP-DL model path')
    parser.add_argument('--no-quantize', action='store_true', help='Disable quantization')
    
    args = parser.parse_args()
    convert_onnx_to_espdl(args.input, args.output, not args.no_quantize)