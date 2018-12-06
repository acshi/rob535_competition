function [t] = sol_time(X, TestTrack, dt)
    pos = [X(1:8:end)'; X(3:8:end)'];
    d = sum((pos - TestTrack.cline(:,end)).^2);
    [~, idx] = min(d);
    
    for i = [idx,idx - 1]
       p0 = pos(:, i); 
       p1 = pos(:, i + 1);
       bl = TestTrack.bl(:, end);
       br = TestTrack.br(:, end);
       q_p = [(p1 - p0) (bl - br)]\(bl - p0);
       
       if q_p(1) > 0 && q_p(2) < norm(p0 - p1) && q_p(2) > 0 && q_p(2) < norm(br - bl)
           t = (i + q_p(1)/norm(p0 - p1))*dt;
           return;
       end
    end
    disp('Error: need more robust sol_time')
    t = nan;
end