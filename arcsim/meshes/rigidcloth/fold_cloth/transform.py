read_file = open('flag.obj', 'r')
write_file = open('vertical_flag.obj', 'w')

for line in read_file:
    if line[:2] != 'v ':
        write_file.write(line)
    else:
        line = line.split()
        write_file.write('v ')
        write_file.write(line[1])
        write_file.write(' 0.502195')
        write_file.write(' '+str(0.622805+float(line[2]))+'\n')

read_file.close()
write_file.close()
